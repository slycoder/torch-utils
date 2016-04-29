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
cmd:option('-min_count', 3, 'Minimum number of word occurrences to include')
cmd:text()

opt = cmd:parse(arg)

local input_file = path.join(opt.data_dir, 'input.txt')

local dataset = {}
local dictionary = {}
local wordCounts = {}
local f = io.open(input_file, 'r')
for line in f:lines() do
  local label = nil
  local words = {}
  for field in string.gmatch(line, '%S+') do
    if label == nil then
      label = field
    else
      if not wordCounts[field] then
        wordCounts[field] = 0
      end
      words[#words + 1] = field
      wordCounts[field] = wordCounts[field] + 1
    end
  end
  dataset[#dataset + 1] = {words, label + 1}
end
f:close()

local filteredDataset = {}
local N = 1
for ii = 1,#dataset do
  local wordsWithWeight = {}
  local nonEmpty = false
  for _, word in ipairs(dataset[ii][1]) do
    if wordCounts[word] >= opt.min_count then
      if not dictionary[word] then
        dictionary[word] = N
        N = N + 1
      end
      wordsWithWeight[#wordsWithWeight + 1] = {dictionary[word], 1.0}
      nonEmpty = true
    end
  end
  if nonEmpty then
    filteredDataset[#filteredDataset + 1] = {
      torch.Tensor(wordsWithWeight),
      dataset[ii][2]
    }
  end
end

print(tablex.size(dictionary))
print(#filteredDataset)

function filteredDataset:size() return #filteredDataset end

torch.save(path.join(opt.data_dir, 'dictionary.t7'), dictionary)
torch.save(path.join(opt.data_dir, 'dataset.t7'), filteredDataset)
