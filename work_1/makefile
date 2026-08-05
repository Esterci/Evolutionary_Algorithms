CXX = g++
CXXFLAGS = -Wall -Wextra -g -O0

TARGET = main
OBJECTS = main.o heuristic.o EVRP.o stats.o

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $(TARGET)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o

heuristic.o: heuristic.cpp
	$(CXX) $(CXXFLAGS) -c heuristic.cpp -o heuristic.o

EVRP.o: EVRP.cpp
	$(CXX) $(CXXFLAGS) -c EVRP.cpp -o EVRP.o

stats.o: stats.cpp
	$(CXX) $(CXXFLAGS) -c stats.cpp -o stats.o

clean:
	rm -f $(OBJECTS) $(TARGET)

rebuild: clean all

.PHONY: all clean rebuild