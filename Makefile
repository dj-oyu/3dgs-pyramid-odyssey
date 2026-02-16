CC  = gcc
CXX = g++

CXXFLAGS = -O3 -march=armv8.2-a+fp16+dotprod -mtune=cortex-a55 \
           -ffast-math -ftree-vectorize -std=c++17 -Wall -Wextra
CFLAGS   = -O3 -march=armv8.2-a+fp16+dotprod -mtune=cortex-a55 \
           -ffast-math -ftree-vectorize -Wall -Wextra

INCLUDES = -Iinclude -Iinclude/ax_sdk
LDFLAGS  = -L/soc/lib -Wl,-rpath,/soc/lib
LIBS     = -lax_sys -lax_vo -lax_ivps -lax_ive -lpthread -lm

SRCDIR   = src
TOOLDIR  = tools
BUILDDIR = build

# Main application sources
SRCS = $(SRCDIR)/main.cpp \
       $(SRCDIR)/gs_scene.cpp \
       $(SRCDIR)/gs_ply_loader.cpp \
       $(SRCDIR)/gs_renderer.cpp \
       $(SRCDIR)/gs_projector.cpp \
       $(SRCDIR)/gs_rasterizer.cpp \
       $(SRCDIR)/gs_camera.cpp \
       $(SRCDIR)/gs_display.cpp \
       $(SRCDIR)/gs_memory.cpp \
       $(SRCDIR)/gs_math.cpp \
       $(SRCDIR)/gs_sort.cpp

OBJS = $(patsubst $(SRCDIR)/%.cpp,$(BUILDDIR)/%.o,$(SRCS))

# Library objects (everything except main)
LIB_OBJS = $(filter-out $(BUILDDIR)/main.o,$(OBJS))

TARGET = gs_splat

# Tool targets
TOOLS = $(BUILDDIR)/vo_test $(BUILDDIR)/ply_info $(BUILDDIR)/vo_diag $(BUILDDIR)/gen_test_ply

.PHONY: all clean tools

all: $(BUILDDIR) $(BUILDDIR)/$(TARGET)

tools: $(BUILDDIR) $(TOOLS)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)
	@echo "Built: $@"

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

# Tools
$(BUILDDIR)/vo_test: $(TOOLDIR)/vo_test.cpp $(BUILDDIR)/gs_memory.o $(BUILDDIR)/gs_display.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $^ $(LIBS)
	@echo "Built: $@"

$(BUILDDIR)/ply_info: $(TOOLDIR)/ply_info.cpp $(BUILDDIR)/gs_ply_loader.o $(BUILDDIR)/gs_memory.o $(BUILDDIR)/gs_scene.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $^ $(LIBS)
	@echo "Built: $@"

$(BUILDDIR)/vo_diag: $(TOOLDIR)/vo_diag.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $^ $(LIBS)
	@echo "Built: $@"

$(BUILDDIR)/gen_test_ply: $(TOOLDIR)/gen_test_ply.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< -lm
	@echo "Built: $@"

clean:
	rm -rf $(BUILDDIR)

# Show compiler info
info:
	@echo "CXX: $(CXX)"
	@$(CXX) --version | head -1
	@echo "CXXFLAGS: $(CXXFLAGS)"
	@echo "Target: $(TARGET)"
