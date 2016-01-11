module DH32StatsHelpers

# package code goes here
export cMDS,
Cohen_d,
DataFrame_to_distmat,
zscore_log10_cols

#cMDS_of_DataFrame,
#kmeans_of_cMDS,

using DataFrames
using Distances
using Distributions
using MultivariateStats
using Clustering

using Distributions
using DataFrames

"""
Compute effect size measure Cohen's d

Input:
  - float array x1
  - float array x2
  - confidence level (default 0.95)

Output:
  - DataFrame with 1 row and 3 columns:
  d, lower_ci, upper_ci
"""
function Cohen_d(x1::Array{Float64,1}, x2::Array{Float64,1}, conf_level::Float64=0.95)
    
    n1 = length(x1)
    n2 = length(x2)
    dof = n1 + n2 - 2.
    d = (mean(x1) - mean(x2))/sqrt((((n1-1.)*var(x1)+(n2-1.)*var(x2))/(n1+n2-2.)))
    S_d = sqrt(((n1 + n2)/(n1 * n2) + 0.5 * d^2/dof) * ((n1 + n2)/dof))
    Z = -quantile(TDist(dof), (1.0-conf_level)/2.0)
    lower_ci = d - Z * S_d
    upper_ci = d + Z * S_d
    
    return DataFrame(d=d, lower_ci=lower_ci, upper_ci=upper_ci)
end


"""
Modify columns of a data frame of count data or other positive data by:

- Replacing 0 by 1 (to avoid NA after log)
- log10-transformation
- z-scores of log10-transformed

Output: transformed data frame
"""
function zscore_log10_cols(df::DataFrame)

    df_new = copy(df)

    for i in 1:length(df)
        
        #replace all zeros by 1 to avoid log(0)
        df_new[df_new[:,i] .== 0,i] = 1

        #take the log10 of all columns
        df_new[:,i] = map(log10,df[:,i])
        
        #take z-scores 
        df_new[:,i] = zscore(convert(Array{Float64},df_new[:,i]))
    end
    
    return df_new
end


"""
Translates a data frame into a distance matrix.

Input:

   - data frame

   - distance (from Distances.jl), e.g. HellingerDist()

   - row_instances: true or false. Are rows of data frame the instances for which distances should be evaluated (true), or the columns (false)

Output: distance matrix (symmetric matrix with zero diagonal)
       
"""
function DataFrame_to_distmat(df::DataFrame, dist::Any, row_instances::Bool=true)

    x = convert(Array, df)

    if (row_instances == true)
        return pairwise(dist, x')
    end

    return pairwise(dist, x)
    
end

"""
Carry out a classical multi-dimensional scaling (cMDS).

Input:

    - distances (symmetric matrix)

    - dimension of cMDS output (default: 2)

    - should eigenvalues be computed? (true or false)

Output:

    - cMDS array with instances as columns

    - if eigenvalues == true: also 1-dim array of eigenvalues
        
"""
function cMDS(d::Array{Float64,2}, dim::Int64=2, eigenvalues::Bool=true)

    #do classical MDS:
    cMDS = classical_mds(d, dim)

    if eigenvalues == true
        #compute eigenvalues
        G = dmat2gram(d)
        E = eigfact!(Symmetric(G))
        #return cMDS array and eigenvalue vector descending from largest
        return cMDS, sort(E.values[:,1],rev=true)
    end
    
    return cMDS

end

end # module
