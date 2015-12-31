-- Perform sparse logistic regression.
-- Data should be preprocessed in a data directory using textToTensor.
-- Fitted model will be saved to the specified model directory.

require 'torch'
require 'nn'
require 'pl'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Run sparse logistic regression')
cmd:text()
cmd:text('Options')
cmd:option('-data_dir', 'data/', 'data directory. Should contain preprocessed data')
cmd:option('-model_dir', 'models/', 'where the models will be written')
cmd:option('-max_iterations', 100, 'the maximum number of iterations of sgd to run')
cmd:text()

opt = cmd:parse(arg)

local dataset = torch.load(path.join(opt.data_dir, 'dataset.t7'));
local dictionary = torch.load(path.join(opt.data_dir, 'dictionary.t7'));
local N = 0
for _, __ in pairs(dictionary) do N = N + 1 end
local model = nn.Sequential()
model:add(nn.SparseLinear(N, 1))
model:add(nn.Sigmoid())

local criterion = nn.BCECriterion()
local trainer = nn.StochasticGradient(model, criterion)
trainer.maxIteration = opt.max_iterations

trainer:train(dataset)
torch.save(
  path.join(
    opt.model_dir,
    string.format('lr_sparse_sgd_i%d.t7', opt.max_iterations)
  ),
  model
)
