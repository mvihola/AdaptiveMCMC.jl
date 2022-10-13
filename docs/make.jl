push!(LOAD_PATH,"../src/")

using Documenter, AdaptiveMCMC

makedocs(sitename="AdaptiveMCMC.jl Documentation", 
pages = [
    "Sampling from log-posteriors" => "index.md",
    "Random walk sampler state" => "rwm.md",
    "Adaptation state" => "adapt.md",
    "Step sizes" => "step.md"
    ]
)

