using Distributed
using LinearAlgebra

# Add worker processes
addprocs(4)

@everywhere using LinearAlgebra

@everywhere function parallel_matrix_multiply(A, B)
    C = A * B
    return C
end

function distribute_multiply(A, B)
    n, m = size(A)
    _, p = size(B)

    if m != size(B, 1)
        error("Matrix dimensions must agree")
    end

    # Split A into chunks
    chunk_size = ceil(Int, n / nworkers())
    A_chunks = [A[(i-1)*chunk_size+1 : min(i*chunk_size, n), :] for i in 1:nworkers()]

    # Distribute the work
    C_chunks = @distributed (vcat) for A_chunk in A_chunks
        parallel_matrix_multiply(A_chunk, B)
    end

    return C_chunks
end

# Initialize matrices A and B
n = 1000
A = randn(n, n)
B = randn(n, n)

# Perform parallel matrix multiplication
@time C = distribute_multiply(A, B)

# Verify the result
@assert isapprox(C, A * B)

println("Matrix multiplication complete and verified!")

# Remove worker processes
rmprocs(workers())
