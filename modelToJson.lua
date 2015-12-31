-- Converts a fitted logistic regression model to

require 'torch'
require 'nn'
require 'json'
require 'pl'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Print out a model as JSON')
cmd:text()
cmd:text('Options')
cmd:option('-data_dir', 'data/', 'data directory. Should contain preprocessed data')
cmd:option('-model', '', 'the filename of the saved model')
cmd:text()

opt = cmd:parse(arg)

-- First flip the dictionary so that we output words instead of indices.
local dictionary = torch.load(path.join(opt.data_dir, 'dictionary.t7'))
local flipped_dictionary = {}
for word, index in pairs(dictionary) do
  flipped_dictionary[index + 1] = word
end

local model = torch.load(opt.model)
local linear_layer = model:get(1)
local output_layer = model:get(3)

local output = {}
output['$BIAS'] = linear_layer.bias[1]

local weight = linear_layer.weight
for i=1,weight:size(2) do
  output[flipped_dictionary[i]] = weight[1][i]
end

print(json.encode(output))
