-- Perform sparse logistic regression.
-- Data should be preprocessed in a data directory using textToTensor.
-- Fitted model will be saved to the specified model directory.

require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'pl'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Run sparse logistic regression')
cmd:text()
cmd:text('Options')
cmd:option('-data_dir', 'data/', 'data directory. Should contain preprocessed data')
cmd:option('-model_dir', 'models/', 'where the models will be written')
cmd:option('-max_iterations', 100, 'the maximum number of iterations of sgd to run')
cmd:option('-batch_size', 10000, 'the size of each batch')
cmd:text()

opt = cmd:parse(arg)

cutorch.setDevice(1)
cutorch.manualSeed(8675309)

local dataset = torch.load(path.join(opt.data_dir, 'dataset.t7'));
local dictionary = torch.load(path.join(opt.data_dir, 'dictionary.t7'));
local N = 0
for _, __ in pairs(dictionary) do N = N + 1 end

local batchSize = opt.batch_size
local numBatches = torch.ceil(dataset:size() / batchSize)
print("dataset size = ", dataset:size())
print("dictionary = ", N)
print("numBatches = ", numBatches)

local model = nn.Sequential()
model:add(nn.Linear(N, 2):cuda())
model:add(nn.LogSoftMax():cuda())

local criterion = nn.ClassNLLCriterion();

model = model:cuda();
criterion = criterion:cuda()

shuffle = torch.randperm(dataset:size())
local bigIdx = 1
for batchIdx=1,numBatches do
  if batchSize > (dataset:size() - bigIdx + 1) then
    batchSize = dataset:size() - bigIdx + 1
  end
  print("Batch", batchIdx, batchSize)
  local batch = torch.FloatTensor(batchSize, N):zero()
  local batchLabels = torch.FloatTensor(batchSize):zero()
  for idx=1,batchSize do
    local datum = dataset[shuffle[bigIdx]][1]
    for j=1,datum:size()[1] do
      batch[idx][datum[j][1]] = 1
    end
    batchLabels[idx] = dataset[shuffle[bigIdx]][2]
    bigIdx = bigIdx + 1
  end
  batch = batch:cuda()
  batchLabels = batchLabels:cuda()

  for i=1,opt.max_iterations do
    local loss = 0

    model:zeroGradParameters()
    local forwarded = model:forward(batch)
    local criterion_forwarded = criterion:forward(forwarded, batchLabels)
    loss = loss + criterion_forwarded
    local criterion_backwarded = criterion:backward(forwarded, batchLabels)
    local gradInput = model:backward(batch, criterion_backwarded)
    model:updateParameters(1.0)
    print(batchIdx, i, loss)
  end
end

torch.save(
  path.join(
    opt.model_dir,
    string.format('lr_sparse_sgd_i%d.t7', opt.max_iterations)
  ),
  model
)
