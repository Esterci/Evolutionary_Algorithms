"""Bridge weight-space DE to the local FisiocomPINN losses and Adam Trainer.

DE directly optimizes network parameters and, optionally, three loss log weights;
it does not train a separate PINN inside each fitness evaluation. The objective
formula and full collocation batches remain fixed, so cached parent fitness is
comparable even though individual genomes encode different loss weights.
"""

import inspect
import math

import torch

from methods.loss_weight_config import loss_log_vars
from methods.pinn_model import Trainer
from fisiocomPinn.Trainer import AdaptiveLossWeights


class PINNFitness:
    """Map genomes to network parameters and optional Initial/Boundary/PDE genes.

    Physical inputs are [N, 3] = (t, x, y); every LOSS has targets [N, 2]
    for (u, v). In adaptive mode the combined PDE MSE remains one loss term,
    and the exact framework objective is sum(exp(-s_i) * L_i + s_i).
    Registered fixed coefficients are ignored, just as in Trainer.adaptive_loop.
    The unweighted sum of raw losses remains a separate reference diagnostic.
    Autograd stays enabled for PDE derivatives; DE retains only scalar fitness.
    """

    def __init__(self, trainer, evolve_loss_weights=False):
        if trainer.adaptive:
            raise ValueError(
                "DE requires loss registration with adaptive=False; "
                "use evolve_loss_weights=True to encode adaptive weights"
            )
        if not trainer.losses or len(trainer.losses) != len(trainer.lossesW):
            raise ValueError("Register losses and their fixed weights before DE")
        if any(not math.isfinite(w) or w <= 0 for w in trainer.lossesW):
            raise ValueError("Loss weights must be finite and positive")
        self.model = trainer.model
        self.losses = tuple(trainer.losses)
        self.evolve_loss_weights = bool(evolve_loss_weights)
        names = [loss.name for loss in self.losses]
        if len(set(names)) != len(names):
            raise ValueError("Each registered loss must have a unique name")
        if self.evolve_loss_weights and names != ["Initial", "Boundary", "PDE"]:
            raise ValueError(
                "Adaptive DE requires Initial, Boundary, PDE losses in that order, "
                "with both PDE residuals grouped in one MSE"
            )
        self.reference_weights = (
            (1.0,) * len(self.losses)
            if self.evolve_loss_weights
            else tuple(float(weight) for weight in trainer.lossesW)
        )
        self.parameters = tuple(p for p in self.model.parameters() if p.requires_grad)
        if not self.parameters:
            raise ValueError("The PINN must have trainable parameters")
        reference = self.parameters[0]
        if any(
            p.device != reference.device or p.dtype != reference.dtype
            for p in self.parameters
        ):
            raise ValueError("All trainable parameters must share device and dtype")
        self.network_dimension = sum(p.numel() for p in self.parameters)
        self._adaptive_weights = None
        if self.evolve_loss_weights:
            self._adaptive_weights = AdaptiveLossWeights(len(self.losses)).to(
                device=reference.device, dtype=reference.dtype
            )
            with torch.no_grad():
                self._adaptive_weights.log_vars.copy_(loss_log_vars(
                    trainer.lossesW, device=reference.device, dtype=reference.dtype))
            # DE varies genes directly and does not backpropagate their fitness.
            self._adaptive_weights.requires_grad_(False)
            self._loss_log_vars = self._adaptive_weights.log_vars
        else:
            self._loss_log_vars = torch.empty(
                0, device=reference.device, dtype=reference.dtype
            )
        self.initial_loss_log_vars = self._loss_log_vars.detach().clone()
        self.dimension = self.network_dimension + self._loss_log_vars.numel()
        self.layout = []
        offset = 0
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                self.layout.append(
                    dict(
                        name=name,
                        shape=list(parameter.shape),
                        start=offset,
                        stop=offset + parameter.numel(),
                    )
                )
                offset += parameter.numel()
        for name in names if self.evolve_loss_weights else ():
            self.layout.append(
                dict(
                    name=f"loss_log_var_{name}",
                    shape=[1],
                    start=offset,
                    stop=offset + 1,
                )
            )
            offset += 1
        self.last_diagnostics = {}

    @property
    def loss_log_vars(self):
        """Return independent Initial/Boundary/PDE genes, or an empty tensor."""
        return self._loss_log_vars.detach().clone()

    @property
    def weights(self):
        """Return exp(-s_i) for adaptive losses or the registered fixed weights."""
        if self.evolve_loss_weights:
            return tuple(
                float(value) for value in torch.exp(-self._loss_log_vars.detach())
            )
        return self.reference_weights

    def bounds(self, lower, upper, log_bound=4.0):
        """Create network bounds and a loss-gene box centered on the initial log coefficients."""
        if (
            not all(math.isfinite(value) for value in (lower, upper, log_bound))
            or lower >= upper
            or log_bound <= 0
        ):
            raise ValueError(
                "Require finite increasing network bounds and positive log_bound"
            )
        reference = self.parameters[0]
        lower_vector = torch.full(
            (self.dimension,), lower, device=reference.device, dtype=reference.dtype
        )
        upper_vector = torch.full_like(lower_vector, upper)
        if self.evolve_loss_weights:
            # Initial samples must have representable positive exp(-s) values.
            if log_bound >= math.log(torch.finfo(reference.dtype).max):
                raise ValueError("log_bound is too large for finite loss multipliers")
            lower_vector[self.network_dimension :] = self.initial_loss_log_vars - log_bound
            upper_vector[self.network_dimension :] = self.initial_loss_log_vars + log_bound
            endpoint_weights = torch.exp(-torch.cat((
                lower_vector[self.network_dimension:], upper_vector[self.network_dimension:])))
            if not torch.isfinite(endpoint_weights).all() or (endpoint_weights <= 0).any():
                raise ValueError("Loss initialization box produces unrepresentable coefficients")
        if (
            not torch.isfinite(lower_vector).all()
            or not torch.isfinite(upper_vector).all()
            or not (lower_vector < upper_vector).all()
        ):
            raise ValueError(
                "Bounds must remain finite and distinct in the model dtype"
            )
        return lower_vector, upper_vector

    def vector(self):
        """Return an independent snapshot of network parameters and log genes."""
        return torch.cat(
            [p.detach().reshape(-1) for p in self.parameters]
            + [self._loss_log_vars.detach()]
        ).clone()

    def _validate_vector_layout(self, vector):
        reference = self.parameters[0]
        if vector.shape != (self.dimension,):
            raise ValueError(f"Expected genome shape ({self.dimension},)")
        if vector.device != reference.device or vector.dtype != reference.dtype:
            raise ValueError("Genome must match the model device and dtype")

    def load(self, vector):
        """Copy a genome without replacing Parameter objects or aliasing DE data."""
        self._validate_vector_layout(vector)
        if not torch.isfinite(vector).all():
            raise ValueError("Network parameters and loss log genes must be finite")
        with torch.no_grad():
            offset = 0
            for parameter in self.parameters:
                count = parameter.numel()
                parameter.copy_(vector[offset : offset + count].reshape_as(parameter))
                offset += count
            self._loss_log_vars.copy_(vector[offset:])

    @torch.enable_grad()
    def terms(self):
        """Evaluate fresh derivative graphs and release each after scalar export."""
        values = {}
        for loss in self.losses:
            value = loss.forward(self.model)
            if value.ndim != 0:
                raise ValueError(f"{loss.name} must produce a scalar loss")
            values[loss.name] = float(value.detach())
            del value
        return values

    def reference_fitness(self, values):
        """Evaluate raw losses with unit adaptive or legacy fixed coefficients."""
        return sum(
            weight * values[loss.name]
            for loss, weight in zip(self.losses, self.reference_weights)
        )

    @torch.no_grad()
    def score(self, values):
        """Use Trainer's exact formula and arithmetic precision in adaptive mode.

        Unbounded finite genes can overflow exp(-s); these candidates receive
        +inf instead of raising a Python exponential overflow exception.
        """
        if self.evolve_loss_weights:
            losses = [
                torch.as_tensor(
                    values[loss.name],
                    device=self._loss_log_vars.device,
                    dtype=self._loss_log_vars.dtype,
                )
                for loss in self.losses
            ]
            total, _ = self._adaptive_weights(losses)
            value = float(total)
        else:
            value = self.reference_fitness(values)
        return value if math.isfinite(value) else math.inf

    def __call__(self, vector):
        self._validate_vector_layout(vector)
        if not torch.isfinite(vector).all():
            self.last_diagnostics = dict(fitness=math.inf, reference_fitness=math.inf)
            return math.inf
        self.load(vector)
        values = self.terms()
        fitness = self.score(values)
        self.last_diagnostics = dict(
            fitness=fitness, reference_fitness=self.reference_fitness(values)
        )
        self.last_diagnostics.update(
            {f"loss_{name}": value for name, value in values.items()}
        )
        self.last_diagnostics.update(
            {
                f"weight_{loss.name}": weight
                for loss, weight in zip(self.losses, self.weights)
            }
        )
        self.last_diagnostics.update(
            {
                f"loss_log_var_{loss.name}": float(value)
                for loss, value in zip(self.losses, self._loss_log_vars)
            }
        )
        return fitness


def make_adam_trainer(
    de_trainer,
    epochs,
    lr=1e-3,
    print_steps=250,
    *,
    loss_log_vars=None,
    betas=(0.9, 0.999),
    training_controls=None,
    optimize_loss_weights=True,
):
    """Continue the selected genome with fresh Adam moments and the same losses.

    The caller first loads DE's best network. Supplying its three loss log genes
    enables Trainer's adaptive loop: Adam then updates both network parameters
    and all three weights. The framework's external-optimizer path preserves a
    supplied AdaptiveLossWeights instance; copy the selected genes there before
    training so they are neither reset to zero nor frozen. Casting that instance
    to the model dtype also avoids the framework's float32 initialization when
    the experiment explicitly requests float64. Raw MSE histories are preserved.
    Omitting loss_log_vars retains the original fixed-weight Adam behavior.
    optimize_loss_weights=False freezes supplied genes while preserving the
    regularized loss formula and raw-sum early-stopping monitor.
    """
    if epochs < 1 or not math.isfinite(lr) or lr <= 0 or print_steps < 1:
        raise ValueError(
            "Adam epochs, learning rate and print interval must be positive"
        )
    if de_trainer.adaptive:
        raise ValueError("DE loss registration must use adaptive=False")
    weights = tuple(de_trainer.lossesW)
    if not weights or len(weights) != len(de_trainer.losses) or any(
        not math.isfinite(weight) or weight <= 0 for weight in weights
    ):
        raise ValueError(
            "Adam needs one finite positive coefficient per registered loss"
        )
    if "optimizer" not in inspect.signature(Trainer).parameters:
        raise RuntimeError(
            "This experiment requires the local Pinn-Torch Trainer "
            "version with the optimizer argument"
        )
    adaptive_weights = None
    parameters = list(de_trainer.model.parameters())
    if loss_log_vars is not None:
        if [loss.name for loss in de_trainer.losses] != ["Initial", "Boundary", "PDE"]:
            raise ValueError("Adaptive Adam requires Initial, Boundary, PDE losses")
        reference = next(parameter for parameter in parameters if parameter.requires_grad)
        log_vars = torch.as_tensor(
            loss_log_vars, device=reference.device, dtype=reference.dtype
        )
        if log_vars.shape != (3,) or not torch.isfinite(log_vars).all():
            raise ValueError("Adam requires three finite loss log genes")
        adaptive_weights = AdaptiveLossWeights(3).to(
            device=reference.device, dtype=reference.dtype
        )
        with torch.no_grad():
            adaptive_weights.log_vars.copy_(log_vars)
        adaptive_weights.requires_grad_(optimize_loss_weights)
        if optimize_loss_weights:
            parameters += list(adaptive_weights.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr, betas=betas)
    from methods.training_controls import trainer_control_kwargs
    control_kwargs = trainer_control_kwargs(optimizer, epochs, training_controls)
    trainer = Trainer(
        n_epochs=epochs,
        model=de_trainer.model,
        device=de_trainer.device,
        adaptive=adaptive_weights is not None,
        optimizer=optimizer,
        lr=lr,
        betas=betas,
        print_steps=print_steps,
        **control_kwargs,
    )
    for loss, weight in zip(de_trainer.losses, weights):
        trainer.add_loss(loss, weight)
    if adaptive_weights is not None:
        trainer.adaptive_weights = adaptive_weights
    return trainer
