BIN    := build/libncnn.a
INCLUDE_DIR := -I./include -I./include/layer -I"../deps/resunpack/include" -I./include/layer/arm -I./include/layer/x86
CXX_SRC_DIR := src
TARGET_DIR := build/targets
OBJ_DIR := build/objs

CC     	:= gcc
CXX		:= g++
LD		:= g++

CFLAGS 	:= $(INCLUDE_DIR) -Wall -O3 -fPIC -fopenmp
CXXFLAGS 	:=  -std=c++11

include ../deps/builder/Makefile.static-library
