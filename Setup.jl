# Include this file, that is,
# julia> include("Setup.jl")
# in order to use the package without installing the package...
function addpath!(this)
  new_path = joinpath(@__DIR__, this)
  if !any(LOAD_PATH .== new_path)
    push!(LOAD_PATH, new_path)
  end
end

addpath!("src")
nothing
