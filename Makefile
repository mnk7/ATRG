CXXFLAGS += -fopenmp -std=c++17 -I./ -march=native
LIBS = -larmadillo -lopenblas

atrg_test: main.o
	$(CXX) -Wall -o build/$@ $(CXXFLAGS) $^ $(LIBS)
	rm -f *.o

atrg_test_debug: CXXFLAGS += -DDEBUG -g
atrg_test_debug: atrg_test

atrg_test_release: CXXFLAGS += -O3
atrg_test_release: atrg_test

clean:
	rm -f build/*
	rm -f *.o
