push!(LOAD_PATH,"../src/")

using Documenter, AdaptiveMCMC

makedocs(sitename = "AdaptiveMCMC.jl Documentation", 
pages = [
    "Introduction" => "index.md",
    "Random walk sampler state" => "rwm.md",
    "Adaptation state" => "adapt.md",
    "Step sizes" => "step.md",
#    "Further examples" => "examples.md"
    ],
#format = Documenter.HTML(prettyurls = false),
)

