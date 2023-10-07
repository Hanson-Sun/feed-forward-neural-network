EXENAME    = ffnn
TESTNAME   = test

BUILD_PATH = ./dist
SOURCE     = ./src
MAIN_FILE  = ./src/main.cpp
TEST_FILE  = ./src/tests.cpp
IDIR       = ./include

# Flags:
CXXFLAGS   = -std=c++20 -c -g -Ofast -Wall -Wextra -pedantic -I$(IDIR)
LDFLAGS    = -std=c++20 
CXX        = g++
LD         = g++

CPP_FILES  += $(filter-out $(TEST_FILE), $(wildcard *.cpp))
CPP_FILES  += $(filter-out $(TEST_FILE), $(wildcard $(SOURCE)/*.cpp))
# OBJ_FILES  += $(addprefix $(BUILD_PATH)/,$(CPP_FILES:.cpp=.o)) 
OBJ_FILES  := $(patsubst $(SOURCE)/%.cpp, $(BUILD_PATH)/%.o, $(CPP_FILES))


# later support for a proper unit testing system
# TEST_CPP   += $(filter-out $(MAIN_FILE), $(CPP_FILES))
# TEST_CPP   += $(TEST_FILE)
# TEST_OBJ    = $(TEST_CPP:.cpp=.o)

all: $(EXENAME) # $(TESTNAME)

$(EXENAME): $(OBJ_FILES)
	$(LD) $(OBJ_FILES) $(LDFLAGS) -o $(BUILD_PATH)/$(EXENAME)

# test: $(TEST_OBJ)
# 	$(LD) $(TEST_OBJ) $(LDFLAGS) -o $(BUILD_PATH)$(TESTNAME)

$(BUILD_PATH)/%.o: $(SOURCE)/%.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	-rm -f *.o $(BUILD_PATH)/*.o $(SOURCE)/*.o