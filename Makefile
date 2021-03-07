EXE=$(addsuffix .exe, $(basename $(wildcard *.cpp)))

CPPC=g++ -std=c++1y
CPPC_FLAGS=-Wall -Wextra -pedantic -Ofast -march=native

all : $(EXE)

%.exe : %.cpp
	$(CPPC) $(CPPC_FLAGS) $< -o$@

clean:
	rm -f $(EXE)

.PHONY: all clean
	
