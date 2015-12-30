-- Preprocesses a text file to a tensor file suitable for
-- usage in most models.  The input file should be one
-- row per input instance, with the label (0 or 1) being
-- in the first position followed by the space separated
-- feature words.
-- The input file is expected to be in data/input.txt
-- and the generated files will be placed there.

require 'torch'
require 'pl'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Preprocess a data source')
cmd:text()
cmd:text('Options')
cmd:option('-data_dir', 'data/', 'data directory. Should contain the file input.txt with input data')
cmd:text()

opt = cmd:parse(arg)

local input_file = path.join(opt.data_dir, 'input.txt')

local untensorized={}
local dictionary = {}
local N = 0
local f = io.open(input_file, 'r')
for line in f:lines() do
  local label = nil
  local words = {}
  for field in string.gmatch(line, '%S+') do
    if label == nil then
      label = field
    else
      if not dictionary[field] then
        dictionary[field] = N
        N = N + 1
      end
      words[#words + 1] = dictionary[field]
    end
  end
  untensorized[#untensorized + 1] = {words, torch.Tensor{label}}
end
f:close()

local dataset = {}
for _, value in ipairs(untensorized) do
  local tensor = torch.zeros(N)
  for _, word in ipairs(value[1]) do
    tensor[word + 1] = 1
  end
  dataset[#dataset + 1] = {tensor, value[2]}
end

function dataset:size() return #dataset end

torch.save(path.join(opt.data_dir, 'dictionary.t7'), dictionary)
torch.save(path.join(opt.data_dir, 'dataset.t7'), dataset)
