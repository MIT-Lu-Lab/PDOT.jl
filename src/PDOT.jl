module PDOT

import LinearAlgebra
import Logging
import Printf
import Random
import StructTypes

using CUDA

const dot = LinearAlgebra.dot
const norm = LinearAlgebra.norm

const ThreadPerBlock = 128
const ThreadPerBlockRow = 32
const ThreadPerBlockCol = 32

include("io.jl")
include("solve_log.jl")
include("termination.jl")
include("utils_gpu.jl")
include("saddle_point_gpu.jl")
include("solver_gpu.jl")

end # module PDOT