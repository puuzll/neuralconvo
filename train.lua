require 'neuralconvo'
require 'xlua'

function zeroDataSize(data)
  -- if type(data) == 'table' then
    -- for i = 1, #data do
      -- data[i] = zeroDataSize(data[i])
    -- end
  if type(data) == 'table' then
    data = {}
  elseif type(data) == 'userdata' then
    data = torch.Tensor():typeAs(data)
  end
  return data
end

function cleanupModel(node)
  if node.output ~= nil then
    node.output = zeroDataSize(node.output)
  end
  if node.gradInput ~= nil then
    node.gradInput = zeroDataSize(node.gradInput)
  end
  if node.finput ~= nil then
    node.finput = zeroDataSize(node.finput)
  end
  if node._gradOutputs ~= nil then
    node._gradOutputs = {}
  end
  if node.outputs ~= nil then
    node.outputs = {}
  end
  if node.sharedClones ~= nil then
    node.sharedClones = {}
  end
  if node.step ~= nil then
    node.step = 1
  end
  -- Recurse on nodes with 'modules'
  if (node.modules ~= nil) then
    if (type(node.modules) == 'table') then
      for i = 1, #node.modules do
        local child = node.modules[i]
        cleanupModel(child)
      end
    end
  end
  node:reset()
  collectgarbage()
end

cmd = torch.CmdLine()
cmd:text('Options:')
cmd:option('--dataset', 0, 'approximate size of dataset to use (0 = all)')
cmd:option('--minWordFreq', 1, 'minimum frequency of words kept in vocab')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--hiddenSize', 300, 'number of hidden units in LSTM')
cmd:option('--learningRate', 0.05, 'learning rate at t=0')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--minLR', 0.00001, 'minimum learning rate')
cmd:option('--saturateEpoch', 20, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--maxEpoch', 50, 'maximum number of epochs to run')
cmd:option('--batchSize', 1000, 'number of examples to load at once')
cmd:option('--layernum', 3, 'num of lstm layer')
cmd:option('--saveIter', 1000, 'num of iteration save model ')

cmd:text()
options = cmd:parse(arg)

print(options)
if options.dataset == 0 then
  -- print("dateset = "..options.dataset)
  options.dataset = nil
end

-- Data
print("-- Loading dataset")
-- dataset = neuralconvo.DataSet(neuralconvo.CornellMovieDialogs("data/cornell_movie_dialogs"),
                    -- {
                      -- dataDir = "data",
                      -- loadFirst = options.dataset,
                      -- minWordFreq = options.minWordFreq
                    -- })

dataset = neuralconvo.NetEaseData({
  dataDir = "SentenceVector",
  loadFirst = options.dataset,
  lexicon = "/lexicon.ult",
  text = "/sentence_sorted_numeric_selected.txt"
})

print("\nDataset stats:")
print("  Vocabulary size: " .. dataset.wordsCount)
print("         Examples: " .. dataset.examplesCount)

-- Model
model = neuralconvo.Seq2Seq(dataset.wordsCount, options.hiddenSize,options.layernum)
model.goToken = dataset.goToken
model.eosToken = dataset.eosToken

-- Training parameters
model.learningRate = options.learningRate
model.momentum = options.momentum
local decayFactor = (options.minLR - options.learningRate) / options.saturateEpoch
local minMeanError = nil

-- print(model.encoder.modules)
-- print(model.encoder.output)
-- print(model.encoder.gradInput)
-- torch.save(dataset.dataDir.."/tmp.t7",model)
-- Enabled CUDA
if options.cuda then
  require 'cutorch'
  require 'cunn'
  model:cuda()
end

-- Run the experiment
-- local maxTimeStep = 40
-- print('try training with batch='..options.batchSize..' maxtimestep='..maxTimeStep)
-- local testInput = torch.ones(options.batchSize,maxTimeStep):cuda()
-- local testDecoder = torch.ones(options.batchSize,maxTimeStep):cuda()
-- local testTarget = torch.ones(options.batchSize,maxTimeStep):cuda()
-- local err = model:train(testInput,testDecoder,testTarget)
print("start fetching ....")
dataset:fetch()
print("end fetching ....")
for epoch = 1, options.maxEpoch do
  print("\n-- Epoch " .. epoch .. " / " .. options.maxEpoch)
  print("")

  local errors = {}
  local timer = torch.Timer()

  local i = 1
  local maxInputLen = 0
  local maxTargetLen = 0
  for examples in dataset:batches(options.batchSize) do

    local encoderInput , decoderInput , decoderTarget= unpack(examples)

    if options.cuda then
      -- input = input:cuda()
      -- target = target:cuda()
      encoderInput = encoderInput:cuda()
      decoderInput = decoderInput:cuda()
      decoderTarget = decoderTarget:cuda()
    end

    if decoderTarget:size(2) > maxTargetLen then
      maxTargetLen = decoderTarget:size(2)
      print("target len = "..maxTargetLen)
    end
    if encoderInput:size(2) > maxInputLen then
      maxInputLen = encoderInput:size(2)
      print("input len = "..maxInputLen)
    end
    local err = model:train(encoderInput,decoderInput,decoderTarget)

    -- Check if error is NaN. If so, it's probably a bug.
    if err ~= err then
      error("Invalid error! Exiting.")
    end

    table.insert(errors , err)
    -- errors[i] = err
    -- xlua.progress(i, options.saveIter)
    i = i + 1
    if i > options.saveIter then
      break
    end
  end

  timer:stop()
  errors = torch.Tensor(errors)

  print("\nFinished in " .. xlua.formatTime(timer:time().real) .. " " .. ( i * options.batchSize / timer:time().real) .. ' examples/sec.')
  print("\nEpoch stats:")
  print("           LR= " .. model.learningRate)
  print("  Errors: min= " .. errors:min())
  print("          max= " .. errors:max())
  print("       median= " .. errors:median()[1])
  print("         mean= " .. errors:mean())
  print("          std= " .. errors:std())

  -- Save the model if it improved.
  if minMeanError == nil or errors:mean() < minMeanError then
    print("\n(Saving model ...)")
    -- model.encoder:get(1).output = torch.Tensor():cuda()
    -- model.encoder:get(1).gradInput = torch.Tensor():cuda()
    -- model.decoder:get(1).output = torch.Tensor():cuda()
    -- model.decoder:get(1).gradInput = torch.Tensor():cuda()
    -- local saveModel = model
    -- local encoder = model.encoder:clone()
    -- local decoder = model.decoder:clone()
    -- cleanupModel(encoder)
    -- cleanupModel(decoder)
    -- saveModel.encoder = encoder
    -- saveModel.decoder = decoder
    -- print(model.encoder.modules)
    -- print(model.encoder.output)
    -- print(model.encoder.gradInput)

    torch.save(dataset.dataDir.."/model.t7", model)
    minMeanError = errors:mean()
  end

  collectgarbage()
  model.learningRate = model.learningRate + decayFactor
  model.learningRate = math.max(options.minLR, model.learningRate)
end

-- Load testing script
require "eval"
