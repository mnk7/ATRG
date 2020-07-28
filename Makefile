CXXFLAGS += -fopenmp -std=c++17 -I./ -I/usr/include/eigen3/
LIBS = -larmadillo -lopenblas -llapack

atrg_test: atrg-test.o
	$(CXX) -Wall -o build/$@ $(CXXFLAGS) $^ $(LIBS)
	rm -f *.o

atrg_test_debug: CXXFLAGS += -DDEBUG -g -Wall
atrg_test_debug: atrg_test

atrg_test_release: CXXFLAGS += -O2
atrg_test_release: atrg_test

clean:
	rm -f build/*
	rm -f *.o
