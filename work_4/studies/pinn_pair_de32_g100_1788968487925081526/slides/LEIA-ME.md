# Slides de resultados

Somente resultados do segundo experimento (32 indivíduos, 100 gerações; três sementes por método).

## Usar no template original

Copie `resultados.tex` e a pasta `figures/` para a pasta do projeto no Overleaf. Insira `\input{resultados.tex}` antes de `\end{document}`. Os pacotes necessários já estão no template fornecido.

## Versão independente

Compile `main.tex` com pdfLaTeX no Overleaf. Se `UoWstyle.sty` e seus recursos estiverem disponíveis, o arquivo utiliza esse tema; caso contrário, utiliza Madrid. O logo UFJF é opcional. O tema e o logo não foram fornecidos no anexo.

São 12 slides de conteúdo, mais a capa na versão independente. As figuras originais foram preservadas, incluindo suas legendas em inglês. As tabelas foram obtidas de `confirmation_summary.json`, com desvio-padrão amostral sobre três sementes. Os campos representam a semente 2028; as tabelas agregam as três sementes.

## Ajustes necessários nos slides anteriores

O template recebido apresenta equações advectivas e condições iniciais diferentes das utilizadas nos testes. Substitua os slides “Sistema resolvido” e “Condição inicial” pela formulação do slide “Problema efetivamente utilizado nos testes”; ele pode ser movido para a seção Métodos.

No slide genômico, F e CR são controles associados ao indivíduo, armazenados separadamente dos 2.309 genes (2.306 parâmetros de rede e três log variâncias).

O termo +s penaliza a redução dos pesos, mas não torna o objetivo limitado inferiormente: para uma perda exatamente zero, s pode tender a menos infinito. O slide final registra essa limitação.

## Verificação

Os caminhos das seis figuras e os ambientes LaTeX foram verificados. Não foi possível compilar localmente: pdfLaTeX e Tectonic não estão instalados. Não há PDF validado neste pacote.
