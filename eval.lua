require 'neuralconvo'
local tokenizer = require "tokenizer"
local list = require "pl.List"
local options = {}

if dataset == nil then
  cmd = torch.CmdLine()
  cmd:text('Options:')
  cmd:option('--cuda', false, 'use CUDA. Training must be done on CUDA')
  cmd:option('--debug', false, 'show debug info')
  cmd:option('--dataDir', 'data', 'show debug info')
  cmd:text()
  options = cmd:parse(arg)

  -- Data
  dataset = neuralconvo.NetEaseData(options)

  -- Enabled CUDA
  if options.cuda then
    require 'cutorch'
    require 'cunn'
  end
end

if model == nil then
  print("-- Loading model")
  model = torch.load(options.dataDir.."/model.t7")
end

-- Word IDs to sentence
function pred2sent(wordIds, i)
  local words = {}
  i = i or 1

  for _, wordId in ipairs(wordIds) do
    local word = dataset.id2word[wordId[i]]
    table.insert(words, word)
  end

  return tokenizer.join(words)
end

function printProbabilityTable(wordIds, probabilities, num)
  print(string.rep("-", num * 22))

  for p, wordId in ipairs(wordIds) do
    local line = "| "
    for i = 1, num do
      local word = dataset.id2word[wordId[i]]
      line = line .. string.format("%-10s(%4d%%)", word, probabilities[p][i] * 100) .. "  |  "
    end
    print(line)
  end

  print(string.rep("-", num * 22))
end

function say(text)
  local wordIds = {}

  local values = stringx.split(text, " ")
  for _,word in ipairs(values) do
    print("w : = "..word)
    local id = dataset.word2id[word] or dataset.unknownToken
    print("id : = "..id)
    table.insert(wordIds, id)
  end

  local input = torch.Tensor(list.reverse(wordIds))
  print(input)
  local wordIds, probabilities = model:eval(input)

  print(wordIds,probabilities)
  print(">> " .. pred2sent(wordIds))

  if options.debug then
    printProbabilityTable(wordIds, probabilities, 4)
  end
end
