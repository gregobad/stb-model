function rep_row(x,n)
	x[ones(Int,n),:]
end

function rep_col(x,n)
	x[:,ones(Int,n)]
end

function read_topics(ch, tz)
	 [CSV.File(string("topic_weights_", chans[i], "_", tz, ".csv"); header=1, type=Float64, drop = [:date, :time_block_15]) |> Tables.matrix for i in 1:length(chans)]
end

function norm_cols(A::Matrix)
	A ./ sum(A,dims=1);
end

function norm_rows(A::Matrix)
	A ./ sum(A,dims=2);
end

function demean_cols(A::Matrix)
	A .- mapslices(Statistics.mean, A, dims=1);
end

function demean_rows(A::Matrix)
	A .- mapslices(Statistics.mean, A, dims=2);
end

function simulate_path(x0::Matrix, innov::Matrix)
	cumsum(innov, dims=2) .+ x0;
end

function index_into!(A_out::Array{T,N1}, A_in::Array{T,N2}, indices::Vararg{Array{Int,N1},N2}) where {T,N1,N2}
	for i in 1:length(A_out)
		A_out[i] = A_in[map(x->x[i], indices)...];
	end
	nothing
end

function index_into!(A_out::Matrix{T}, A_in::Matrix{T}, indices::Vararg{Array{Int,2},2}) where {T}
	i_1, i_2 = indices;
	for i in 1:length(A_out)
		A_out[i] = A_in[i_1[i],i_2[i]];
	end
	nothing
end

function index_into!(A_out::Matrix{T}, A_in::Array{T,3}, indices::Vararg{Array{Int,2},3}) where {T}
	i_1, i_2, i_3 = indices;
	for i in 1:length(A_out)
		A_out[i] = A_in[i_1[i],i_2[i],i_3[i]];
	end
	nothing
end


function abs!(x :: AbstractVector)
	for i in length(x)
		x[i] = abs(x[i])
	end
end
