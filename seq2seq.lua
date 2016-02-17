-- Based on https://github.com/Element-Research/rnn/blob/master/examples/encoder-decoder-coupling.lua
local Seq2Seq = torch.class("neuralconvo.Seq2Seq")

function Seq2Seq:__init(vocabSize, hiddenSize, layernum)
  self.vocabSize = assert(vocabSize, "vocabSize required at arg #1")
  self.hiddenSize = assert(hiddenSize, "hiddenSize required at arg #2")
  if nil == layernum then
    layernum = 1
  end
  self.layernum = layernum
  self:buildModel()
end

function Seq2Seq:buildModel()
  self.encoder = nn.Sequential()
  self.encoder:add(nn.LookupTableMaskZero(self.vocabSize, self.hiddenSize))
  self.encoder:add(nn.SplitTable(1, 2))

  self.encoderLSTM = {}
  local encoderLayer = nn.Sequential()
  for i = 1,self.layernum do
    local lstm = nn.GRU(self.hiddenSize, self.hiddenSize)
    table.insert(self.encoderLSTM,lstm)
    encoderLayer:add(lstm)
  end
  self.encoder:add(nn.Sequencer(nn.MaskZero(encoderLayer,1)))
  -- self.encoderLSTM = nn.FastLSTM(self.hiddenSize, self.hiddenSize)
  -- -- self.encoderLSTM.usenngraph = true
  -- self.encoder:add(nn.Sequencer(nn.MaskZero(self.encoderLSTM,1)))
  self.encoder:add(nn.SelectTable(-1))

  self.decoder = nn.Sequential()
  self.decoder:add(nn.LookupTableMaskZero(self.vocabSize, self.hiddenSize))
  self.decoder:add(nn.SplitTable(1, 2))
  -- self.decoderLSTM = nn.FastLSTM(self.hiddenSize, self.hiddenSize)
  -- -- self.decoderLSTM.usenngraph = true
  -- self.decoderLayer = nn.Sequential():add(self.decoderLSTM):add(nn.Linear(self.hiddenSize,self.vocabSize)):add(nn.LogSoftMax())
  -- self.decoder:add(nn.Sequencer(nn.MaskZero(self.decoderLSTM,1)))
  -- self.decoder:add(nn.Sequencer(nn.MaskZero(nn.Linear(self.hiddenSize, self.vocabSize),1)))
  -- self.decoder:add(nn.Sequencer(nn.MaskZero(nn.LogSoftMax(),1)))

  self.decoderLSTM = {} 
  self.decoderLayer = nn.Sequential()
  for i = 1,self.layernum do
    local lstm = nn.GRU(self.hiddenSize, self.hiddenSize)
    table.insert(self.decoderLSTM,lstm)
    self.decoderLayer:add(lstm)
  end
  self.decoderLayer:add(nn.Linear(self.hiddenSize,self.vocabSize)):add(nn.LogSoftMax())
  self.decoder:add(nn.Sequencer(nn.MaskZero(self.decoderLayer,1)))

  self.encoder:zeroGradParameters()
  self.decoder:zeroGradParameters()

  self.zeroTensor = torch.Tensor(2):zero()
  self.criterion = nn.MaskZeroCriterion(nn.ClassNLLCriterion(),1)
  print('encoder:')
  for i,module in ipairs(self.encoder:listModules()) do
    print(module)
    break
  end
  print('decoder:')
  for i,module in ipairs(self.decoder:listModules()) do
    print(module)
    break
  end
  self.encoder:training()
  self.decoder:training()
end

function Seq2Seq:cuda()
  self.encoder:cuda()
  self.decoder:cuda()

  if self.criterion then
    self.criterion:cuda()
  end

  self.zeroTensor = self.zeroTensor:cuda()
end

--[[ Forward coupling: Copy encoder cell and output to decoder LSTM ]]--
function Seq2Seq:forwardConnect(inputSeqLen)
  -- self.decoderLSTM.userPrevOutput =
    -- nn.rnn.recursiveCopy(self.decoderLSTM.userPrevOutput, self.encoderLSTM.outputs[inputSeqLen])
  -- self.decoderLSTM.userPrevCell =
    -- nn.rnn.recursiveCopy(self.decoderLSTM.userPrevCell, self.encoderLSTM.cells[inputSeqLen])
    for i = 1,self.layernum do
      self.decoderLSTM[i].userPrevOutput =
      nn.rnn.recursiveCopy(self.decoderLSTM[i].userPrevOutput, self.encoderLSTM[self.layernum + 1 -i].outputs[inputSeqLen])
      -- self.decoderLSTM[i].userPrevCell =
      -- nn.rnn.recursiveCopy(self.decoderLSTM[i].userPrevCell, self.encoderLSTM[self.layernum + 1 -i].cells[inputSeqLen])
    end
end

--[[ Backward coupling: Copy decoder gradients to encoder LSTM ]]--
function Seq2Seq:backwardConnect()
  -- self.encoderLSTM.userNextGradCell =
  -- nn.rnn.recursiveCopy(self.encoderLSTM.userNextGradCell, self.decoderLSTM.userGradPrevCell)
  -- self.encoderLSTM.gradPrevOutput =
  -- nn.rnn.recursiveCopy(self.encoderLSTM.gradPrevOutput, self.decoderLSTM.userGradPrevOutput)
  for i = 1,self.layernum do
    -- self.encoderLSTM[self.layernum - i + 1].userNextGradCell =
    -- nn.rnn.recursiveCopy(self.encoderLSTM[self.layernum - i + 1].userNextGradCell, self.decoderLSTM[i].userGradPrevCell)
    self.encoderLSTM.gradPrevOutput =
    nn.rnn.recursiveCopy(self.encoderLSTM[self.layernum - i + 1].gradPrevOutput, self.decoderLSTM[i].userGradPrevOutput)
  end
end

function Seq2Seq:train(encoderInput,decoderInput,decoderTarget)

  --[[ only batch processing ]]
  local timeStep = decoderTarget:size(2)
  -- print("decoderTarget : ")
  -- print(decoderTarget)
  decoderTarget = nn.SplitTable(2):forward(decoderTarget)
  -- Forward pass
  -- print("encoderInput : ")
  -- print(encoderInput)
  self.encoder:forward(encoderInput)
  self:forwardConnect(encoderInput:size(2))
  local decoderOutput = self.decoder:forward(decoderInput)
  -- print("decoderInput :")
  -- print(decoderInput)
  -- print("decoderOutput :")
  -- print(decoderOutput)
  -- print("decoderOutput :")
  -- print(decoderOutput[4]:sub(1,-1,1,5))
  -- print(decoderOutput[4])
  -- print("decoderTarget:")
  -- print(decoderTarget)
  -- print(decoderTarget[4])
  local Edecoder = 0
  local gEdec = {}
  for i = 1,timeStep do 
    Edecoder = Edecoder + self.criterion:forward(decoderOutput[i],decoderTarget[i])
    gEdec[i] = self.criterion:backward(decoderOutput[i],decoderTarget[i]):clone()
  end
  -- print(Edecoder)


  -- local Edecoder = self.criterion:forward(decoderOutput, decoderTarget)
  -- if Edecoder ~= Edecoder then -- Exist early on bad error
    -- return Edecoder
  -- end
  -- -- Backward pass
  -- local gEdec = self.criterion:backward(decoderOutput, decoderTarget)

  -- print("gEdec :")
  -- print(gEdec)
  -- -- print(gEdec[4]:sub(1,-1,1,5))
  -- print(gEdec[4])
  self.decoder:backward(decoderInput, gEdec)
  self:backwardConnect()
  self.encoder:backward(encoderInput, self.zeroTensor)

  self.encoder:updateGradParameters(self.momentum)
  self.decoder:updateGradParameters(self.momentum)
  self.decoder:updateParameters(self.learningRate)
  self.encoder:updateParameters(self.learningRate)
  self.encoder:zeroGradParameters()
  self.decoder:zeroGradParameters()

  self.decoder:forget()
  self.encoder:forget()

  return Edecoder
end

local MAX_OUTPUT_SIZE = 20

function Seq2Seq:eval(input)
  assert(self.goToken, "No goToken specified")
  assert(self.eosToken, "No eosToken specified")

  -- print("input = ")
  -- print(input)

  self.encoder:forward(input)
  self:forwardConnect(input:size(1))

  local predictions = {}
  local probabilities = {}

  -- Forward <go> and all of it's output recursively back to the decoder
  local output = self.goToken
  for i = 1, MAX_OUTPUT_SIZE do
    -- print("output = "..output)
    local prediction = self.decoder:forward(torch.Tensor{output})[1]
    -- prediction contains the probabilities for each word IDs.
    -- The index of the probability is the word ID.
    -- print ("prediction = ")
    -- print(prediction)
    local prob, wordIds = prediction:sort(1, true)

    -- First one is the most likely.
    output = wordIds[1]

    -- Terminate on EOS token
    if output == self.eosToken then
      break
    end

    table.insert(predictions, wordIds)
    table.insert(probabilities, prob)
  end 

  self.decoder:forget()
  self.encoder:forget()

  return predictions, probabilities
end
