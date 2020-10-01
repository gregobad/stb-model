@everywhere import ArrayInterface
@everywhere import FiniteDiff
@everywhere import SharedArrays
@everywhere import Optim
@everywhere import LineSearches
@everywhere import Logging

Logging.disable_logging(Logging.LogLevel(3));

function parallel_jacobian!(
    J::SharedArrays.SharedArray{returntype,2},
    f,
    x,
    cache::FiniteDiff.JacobianCache{T1,T2,T3,cType,sType,fdtype,returntype},
    f_in = nothing;
    relstep = FiniteDiff.default_relstep(fdtype, eltype(x)),
    absstep = relstep,
    colorvec = cache.colorvec,
    sparsity = cache.sparsity,
    dir = true) where {T1,T2,T3,cType,sType,fdtype,returntype}

    m, n = size(J)
    _color = reshape(colorvec, axes(x)...)

    x1, fx, fx1 = cache.x1, cache.fx, cache.fx1
    copyto!(x1, x)
    vfx = FiniteDiff._vec(fx)

    rows_index = nothing
    cols_index = nothing
    if FiniteDiff._use_findstructralnz(sparsity)
        rows_index, cols_index = ArrayInterface.findstructralnz(sparsity)
    end

    if sparsity !== nothing
        fill!(J,false)
    end


    if fdtype == Val{:forward}
        vfx1 = FiniteDiff._vec(fx1)

        if f_in isa Nothing
            f(fx, x)
            vfx = FiniteDiff._vec(fx)
        else
            vfx = FiniteDiff._vec(f_in)
        end

        @sync @distributed for color_i ∈ 1:maximum(colorvec)
            if sparsity isa Nothing
                x1_save = ArrayInterface.allowed_getindex(x1,color_i)
                epsilon = FiniteDiff.FiniteDiff.compute_epsilon(Val{:forward}, x1_save, relstep, absstep, dir)
                ArrayInterface.allowed_setindex!(x1, x1_save + epsilon, color_i)
                f(fx1, x1)
                # J is dense, so either it is truly dense or this is the
                # compressed form of the coloring, so write into it.
                @. J[:,color_i] = (vfx1 - vfx) / epsilon
                # Now return x1 back to its original value
                ArrayInterface.allowed_setindex!(x1, x1_save, color_i)
            else # Perturb along the colorvec vector
                @. fx1 = x1 * (_color == color_i)
                tmp = norm(fx1)
                epsilon = FiniteDiff.compute_epsilon(Val{:forward}, sqrt(tmp), relstep, absstep, dir)
                @. x1 = x1 + epsilon * (_color == color_i)
                f(fx1, x1)
                # J is a sparse matrix, so decompress on the fly
                @. vfx1 = (vfx1 - vfx) / epsilon
                if ArrayInterface.fast_scalar_indexing(x1)
                    FiniteDiff._colorediteration!(J,sparsity,rows_index,cols_index,vfx1,colorvec,color_i,n)
                else
                    #=
                    J.nzval[rows_index] .+= (colorvec[cols_index] .== color_i) .* vfx1[rows_index]
                    or
                    J[rows_index, cols_index] .+= (colorvec[cols_index] .== color_i) .* vfx1[rows_index]
                    += means requires a zero'd out start
                    =#
                    if J isa SparseMatrixCSC
                        @. void_setindex!((J.nzval,), getindex((J.nzval,), rows_index) + (getindex((_color,), cols_index) == color_i) * getindex((vfx1,), rows_index), rows_index)
                    else
                        @. void_setindex!((J,), getindex((J,), rows_index, cols_index) + (getindex((_color,), cols_index) == color_i) * getindex((vfx1,), rows_index), rows_index, cols_index)
                    end
                end
                # Now return x1 back to its original value
                @. x1 = x1 - epsilon * (_color == color_i)
            end
        end #for ends here
    elseif fdtype == Val{:central}
        vfx1 = FiniteDiff._vec(fx1)
        @sync @distributed for color_i ∈ 1:maximum(colorvec)
            if sparsity isa Nothing
                x_save = ArrayInterface.allowed_getindex(x, color_i)
                epsilon = FiniteDiff.compute_epsilon(Val{:central}, x_save, relstep, absstep, dir)
                ArrayInterface.allowed_setindex!(x1, x_save + epsilon, color_i)
                f(fx1, x1)
                ArrayInterface.allowed_setindex!(x1, x_save - epsilon, color_i)
                f(fx, x1)
                @. J[:,color_i] = (vfx1 - vfx) / 2epsilon
                ArrayInterface.allowed_setindex!(x1, x_save, color_i)
            else # Perturb along the colorvec vector
                @. fx1 = x1 * (_color == color_i)
                tmp = norm(fx1)
                epsilon = FiniteDiff.compute_epsilon(Val{:central}, sqrt(tmp), relstep, absstep, dir)
                @. x1 = x1 + epsilon * (_color == color_i)
                @. x  = x  - epsilon * (_color == color_i)
                f(fx1, x1)
                f(fx, x)
                @. vfx1 = (vfx1 - vfx) / 2epsilon
                if ArrayInterface.fast_scalar_indexing(x1)
                    FiniteDiff._colorediteration!(J,sparsity,rows_index,cols_index,vfx1,colorvec,color_i,n)
                else
                    if J isa SparseMatrixCSC
                        @. void_setindex!((J.nzval,), getindex((J.nzval,), rows_index) + (getindex((_color,), cols_index) == color_i) * getindex((vfx1,), rows_index), rows_index)
                    else
                        @. void_setindex!((J,), getindex((J,), rows_index, cols_index) + (getindex((_color,), cols_index) == color_i) * getindex((vfx1,), rows_index), rows_index, cols_index)
                    end
                end
                @. x1 = x1 - epsilon * (_color == color_i)
                @. x  = x  + epsilon * (_color == color_i)
            end
        end
    elseif fdtype==Val{:complex} && returntype<:Real
        epsilon = FiniteDiff.eps(eltype(x))
        @distributed for color_i ∈ 1:maximum(colorvec)
            if sparsity isa Nothing
                x1_save = ArrayInterface.allowed_getindex(x1, color_i)
                ArrayInterface.allowed_setindex!(x1, x1_save + im*epsilon, color_i)
                f(fx,x1)
                @. J[:,color_i] = imag(vfx) / epsilon
                ArrayInterface.allowed_setindex!(x1, x1_save,color_i)
            else # Perturb along the colorvec vector
                @. x1 = x1 + im * epsilon * (_color == color_i)
                f(fx,x1)
                @. vfx = imag(vfx) / epsilon
                if ArrayInterface.fast_scalar_indexing(x1)
                    FiniteDiff._colorediteration!(J,sparsity,rows_index,cols_index,vfx,colorvec,color_i,n)
                else
                   if J isa SparseMatrixCSC
                        @. void_setindex!((J.nzval,), getindex((J.nzval,), rows_index) + (getindex((_color,), cols_index) == color_i) * getindex((vfx,),rows_index), rows_index)
                    else
                        @. void_setindex!((J,), getindex((J,), rows_index, cols_index) + (getindex((_color,), cols_index) == color_i) * getindex((vfx,), rows_index), rows_index, cols_index)
                    end
                end
                @. x1 = x1 - im * epsilon * (_color == color_i)
            end
        end
    else
        FiniteDiff.fdtype_error(returntype)
    end
    nothing
end


function parallel_jacobian(
    f,
    x,
    cache::FiniteDiff.JacobianCache{T1,T2,T3,cType,sType,fdtype,returntype},
    f_in = nothing;
    relstep = FiniteDiff.default_relstep(fdtype, eltype(x)),
    absstep = relstep,
    colorvec = cache.colorvec,
    sparsity = cache.sparsity,
    dir = true) where {T1,T2,T3,cType,sType,fdtype,returntype}


    if !(sparsity == nothing)
        FiniteDiff.fdtype_error(returntype)
    end

    _color = reshape(colorvec, axes(x)...)

    x1, fx, fx1 = cache.x1, cache.fx, cache.fx1
    copyto!(x1, x)
    vfx = FiniteDiff._vec(fx)

    rows_index = nothing
    cols_index = nothing
    if FiniteDiff._use_findstructralnz(sparsity)
        rows_index, cols_index = ArrayInterface.findstructralnz(sparsity)
    end

    if fdtype == Val{:forward}
        vfx1 = FiniteDiff._vec(fx1)

        if f_in isa Nothing
            f(fx, x)
            vfx = FiniteDiff._vec(fx)
        else
            vfx = FiniteDiff._vec(f_in)
        end

        J = @distributed (hcat) for color_i ∈ 1:maximum(colorvec)
            x1_save = ArrayInterface.allowed_getindex(x1,color_i)
            epsilon = FiniteDiff.FiniteDiff.compute_epsilon(Val{:forward}, x1_save, relstep, absstep, dir)
            ArrayInterface.allowed_setindex!(x1, x1_save + epsilon, color_i)
            f(fx1, x1)
            # Now return x1 back to its original value
            ArrayInterface.allowed_setindex!(x1, x1_save, color_i)
            @. (vfx1 - vfx) / epsilon
        end  #for ends here

    elseif fdtype == Val{:central}
        vfx1 = FiniteDiff._vec(fx1)
        J = @distributed (hcat) for color_i ∈ 1:maximum(colorvec)
            x_save = ArrayInterface.allowed_getindex(x, color_i)
            epsilon = FiniteDiff.compute_epsilon(Val{:central}, x_save, relstep, absstep, dir)
            ArrayInterface.allowed_setindex!(x1, x_save + epsilon, color_i)
            f(fx1, x1)
            ArrayInterface.allowed_setindex!(x1, x_save - epsilon, color_i)
            f(fx, x1)
            ArrayInterface.allowed_setindex!(x1, x_save, color_i)
            @. (vfx1 - vfx) / (2*epsilon)
        end
    elseif fdtype==Val{:complex} && returntype<:Real
        epsilon = FiniteDiff.eps(eltype(x))
        J = @distributed (hcat) for color_i ∈ 1:maximum(colorvec)
            x1_save = ArrayInterface.allowed_getindex(x1, color_i)
            ArrayInterface.allowed_setindex!(x1, x1_save + im*epsilon, color_i)
            f(fx,x1)
            ArrayInterface.allowed_setindex!(x1, x1_save,color_i)
            @. imag(vfx) / epsilon
        end
    else
        FiniteDiff.fdtype_error(returntype)
    end

    J


end
