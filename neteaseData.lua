--[[
Format movie dialog data as a table of line 1:

  { {word_ids of character1}, {word_ids of character2} }

Then flips it around and get the dialog from the other character's perspective:

  { {word_ids of character2}, {word_ids of character1} }

Also builds the vocabulary.
]]-- 

local NetEaseData= torch.class("neuralconvo.NetEaseData")
local xlua = require "xlua"
local list = require "pl.List"

local function parsedLines(file)
  local f = assert(io.open(file, 'r'))

  return function()
    local line = f:read("*line")

    if line == nil then
      f:close()
      return
    end

    return line
  end
end

function NetEaseData:__init(options)
  self.options = options or {}

  self.dataDir = options.dataDir
  self.examplesFilename = self.dataDir.."/examples.t7"

  -- Discard words with lower frequency then this
  self.minWordFreq = options.minWordFreq or 1

  -- Maximum number of words in an example sentence
  self.maxExampleLen = options.maxExampleLen or 25

  -- Load only first fews examples (approximately)
  self.loadFirst = options.loadFirst

  -- self.examples = {}
  self.word2id = {}
  self.id2word = {}

  self:load(options)
end


function NetEaseData:loadLexicon(filename)
  print(filename)
  local idx = 1
  for line in parsedLines(filename) do
    -- print(line)
    self.word2id[line] = idx
    self.id2word[idx] = line
    idx = 1 + idx
  end
  print("A idx : "..self.word2id["A"])
  print("frist idx : "..self.id2word[1])
end

function NetEaseData:load(options)
  local filename = options.dataDir.."/vocab.t7"

  if path.exists(filename) then
    print("Loading vocabulary from " .. filename .. " ...")
    local data = torch.load(filename)
    self.word2id = data.word2id
    self.id2word = data.id2word
    self.wordsCount = data.wordsCount
    self.goToken = data.goToken
    self.eosToken = data.eosToken
    self.unknownToken = data.unknownToken
    self.examplesCount = data.examplesCount
  else
    print("" .. filename .. " not found")
    -- self:visit(loader:load())
    self:loadLexicon(options.dataDir..options.lexicon)
    self:loadText(options.dataDir..options.text)
    self.wordsCount = #self.id2word
    print('wordcount = '..self.wordsCount)
    print("Writing " .. filename .. " ...")
    torch.save(filename, {
      word2id = self.word2id,
      id2word = self.id2word,
      wordsCount = self.wordsCount,
      goToken = self.goToken,
      eosToken = self.eosToken,
      unknownToken = self.unknownToken,
      examplesCount = self.examplesCount
    })
  end
end

function NetEaseData:loadText(filename)
  -- Table for keeping track of word frequency
  -- self.wordFreq = {}
  -- self.examples = {}

  print("-- Pre-processing data")
  -- Add magic tokens
  self.unknownToken = self.word2id['<UNK>'] -- Word dropped from vocabulary
  self.goToken = "<go>" -- Start of sequence
  self.eosToken = "<eos>" -- End of sequence
  table.insert(self.id2word,self.goToken)
  self.goIdx = table.maxn(self.id2word)
  self.word2id[self.goToken] = self.goIdx 
  table.insert(self.id2word,self.eosToken)
  self.eosIdx = table.maxn(self.id2word)
  self.word2id[self.eosToken] = self.eosIdx
  self.examplesCount = 0
  print(self.word2id[1])
  print(self.id2word[1])
  self.goToken = self.goIdx
  self.eosToken = self.eosIdx
  collectgarbage()
  local file = torch.DiskFile(self.examplesFilename, "w")
  file:referenced(false)
  for line in parsedLines(filename) do
    local values = stringx.split(line, "\t")
    local input = torch.Tensor(values)
    local target = values
    table.insert(target,1,self.goIdx)
    -- for i, wordId in ipairs(values) do
      -- wordId = tonumber(wordId)
      -- table.insert(input,wordId)
      -- table.insert(target,wordId)
    -- end
    table.insert(target,self.eosIdx)
    -- print(input)
    -- print(target)
    -- table.insert(self.examples,})
    file:writeObject({input,torch.Tensor(target)})
    self.examplesCount = self.examplesCount + 1  
    if self.examplesCount % 10000  == 0 then
      xlua.progress(self.examplesCount, 4143059)
      -- collectgarbage()
    end

    if self.loadFirst and self.examplesCount > self.loadFirst then
      break
    end
    -- for i, example in ipairs(self.examples) do
      -- xlua.progress(i, #self.examples)
    -- end

  end

  file:close()

  -- local total = self.loadFirst or #conversations * 2

  -- for i, conversation in ipairs(conversations) do
    -- if i > total then break end
    -- self:visitConversation(conversation)
    -- xlua.progress(i, total)
  -- end


  -- print("-- Removing low frequency words")

  -- for i, datum in ipairs(self.examples) do
    -- self:removeLowFreqWords(datum[1])
    -- self:removeLowFreqWords(datum[2])
    -- xlua.progress(i, #self.examples)
  -- end

  -- self.wordFreq = nil

  -- self.examplesCount = #self.examples
  -- self:writeExamplesToFile()
  -- self.examples = nil

  collectgarbage()
end

-- function NetEaseData:writeExamplesToFile()
  -- print("Writing " .. self.examplesFilename .. " ...")
  -- local file = torch.DiskFile(self.examplesFilename, "w")
  -- file:referenced(false)

  -- for i, example in ipairs(self.examples) do
    -- file:writeObject(example)
    -- xlua.progress(i, #self.examples)
  -- end

  -- file:close()
-- end

function NetEaseData:fetch()
  print(collectgarbage("count"))
  self.inputList = {}
  self.outoutList = {}
  local file = torch.DiskFile(self.examplesFilename, "r")
  file:referenced(false)
  file:quiet()
  while(true) do
    local example = file:readObject()
    if example == nil then
      -- done = true
      file:close()
      -- return examples
      break
    end
    local input, target = unpack(example)
    table.insert(self.inputList,input)
    table.insert(self.outoutList,target)
    -- table.insert(self.examples,{input:storage(),target:storage()})
    -- table.insert(self.examples,example)
    if #self.inputList % 1000000 == 0 then
      print("fetching..."..#self.inputList)
      collectgarbage()
      print(collectgarbage("count"))
    end
  end
  collectgarbage()
  print(collectgarbage("count"))
end

function NetEaseData:batches(size)
  -- local file = torch.DiskFile(self.examplesFilename, "r")
  -- file:referenced(false)
  -- file:quiet()
  -- local done = false

  return function()
    -- if done then
      -- return
    -- end

    local start = torch.random(self.examplesCount)
    local examples = {}
    local maxInputLen = 0
    local maxOutputLen = 0
    for i = 1, size do
      local pos = start + i -1
      -- if example == nil then
        -- done = true
        -- file:close()
        -- -- return examples
        -- break
      -- end
      if pos > self.examplesCount then 
        break
      end
      -- local example = file:readObject()
      local example = {self.inputList[pos],self.outoutList[pos]}
      local input, target = unpack(example)
      -- print(input)
      -- input = torch.Tensor(input)
      -- target = torch.Tensor(target)
      if input:size(1) > maxInputLen then
          maxInputLen = input:size(1)
      end
      if target:size(1) > maxOutputLen then
          maxOutputLen = target:size(1)
      end
      table.insert(examples, example)
    end
    local sampleNum = #examples
    local encoderInput = torch.IntTensor(sampleNum,maxInputLen):zero()
    local decoderInput = torch.IntTensor(sampleNum,maxOutputLen-1):zero()
    local decoderTarget = torch.IntTensor(sampleNum,maxOutputLen-1):zero()
    for idx, example in ipairs(examples) do
        local input, target = unpack(example)
        encoderInput:sub(idx,idx,-input:size(1),-1):copy(input)
        -- print(decoderInput:size())
        -- print(target)
        decoderInput:sub(idx,idx,1,target:size(1)-1):copy(target:sub(1,-2))
        decoderTarget:sub(idx,idx,1,target:size(1)-1):copy(target:sub(2,-1))
    end
    -- print(encoderInput)
    -- print(decoderInput)
    return {encoderInput,decoderInput,decoderTarget}
  end
end

function NetEaseData:removeLowFreqWords(input)
  for i = 1, input:size(1) do
    local id = input[i]
    local word = self.id2word[id]

    if word == nil then
      -- Already removed
      input[i] = self.unknownToken

    elseif self.wordFreq[word] < self.minWordFreq then
      input[i] = self.unknownToken
      
      self.word2id[word] = nil
      self.id2word[id] = nil
      self.wordsCount = self.wordsCount - 1
    end
  end
end

-- function NetEaseData:visitConversation(lines, start)
  -- start = start or 1

  -- for i = start, #lines, 2 do
    -- local input = lines[i]
    -- local target = lines[i+1]

    -- if target then
      -- local inputIds = self:visitText(input.text)
      -- local targetIds = self:visitText(target.text, 2)

      -- if inputIds and targetIds then
        -- -- Revert inputs
        -- inputIds = list.reverse(inputIds)

        -- table.insert(targetIds, 1, self.goToken)
        -- table.insert(targetIds, self.eosToken)

        -- table.insert(self.examples, { torch.IntTensor(inputIds), torch.IntTensor(targetIds) })
      -- end
    -- end
  -- end
-- end

-- function NetEaseData:visitText(text, additionalTokens)
  -- local words = {}
  -- additionalTokens = additionalTokens or 0

  -- if text == "" then
    -- return
  -- end
  
  -- for t, word in tokenizer.tokenize(text) do
    -- table.insert(words, self:makeWordId(word))
    -- -- Only keep the first sentence
    -- if t == "endpunct" or #words >= self.maxExampleLen - additionalTokens then
      -- break
    -- end
  -- end

  -- if #words == 0 then
    -- return
  -- end

  -- return words
-- end

-- function NetEaseData:makeWordId(word)
  -- word = word:lower()

  -- local id = self.word2id[word]

  -- if id then
    -- self.wordFreq[wNetEaseData] = self.wordFreq[word] + 1
  -- else
    -- self.wordsCount = self.wordsCount + 1
    -- id = self.wordsCount
    -- self.id2word[id] = word
    -- self.word2id[word] = id
    -- self.wordFreq[word] = 1
  -- end

  -- return id
-- end
