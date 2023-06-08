const express = require('express');
const cors = require('cors');
const app = express();
app.use(cors());
app.use(express.json());

const nltk = require('nltk');
const WordNetLemmatizer = nltk.WordNetLemmatizer;
const lemmatizer = new WordNetLemmatizer();

const fs = require('fs');
const pickle = require('pickle');
const model = require('@tensorflow/tfjs-node').loadModel('file://chatbot_model.h5');

const intentsRaw = fs.readFileSync('intents.json');
const intents = JSON.parse(intentsRaw);
const words = pickle.load(fs.readFileSync('words.pkl', 'utf-8'));
const classes = pickle.load(fs.readFileSync('classes.pkl', 'utf-8'));

const wikipedia = require('wikipedia');
const moment = require('moment');

function cleanUpSentence(sentence) {
  const sentenceWords = nltk.word_tokenize(sentence);
  const cleanedWords = sentenceWords.map(word => lemmatizer.lemmatize(word.toLowerCase()));
  return cleanedWords;
}

function bagOfWords(sentence, words, showDetails=true) {
  const sentenceWords = cleanUpSentence(sentence);
  const bag = Array(words.length).fill(0);
  for (const s of sentenceWords) {
    for (const [i, w] of words.entries()) {
      if (w === s) {
        bag[i] = 1;
        if (showDetails) {
          console.log(`found in bag: ${w}`);
        }
      }
    }
  }
  return new Float32Array(bag);
}

function predictClass(sentence, model) {
  const p = bagOfWords(sentence, words, false);
  const resRaw = model.predict(p.reshape([1, p.length]));
  const res = resRaw.arraySync()[0];
  const errorThreshold = 0.25;
  const results = res.map((r, i) => ({intent: classes[i], probability: r}))
    .filter(r => r.probability > errorThreshold)
    .sort((a, b) => b.probability - a.probability);
  return results;
}

async function getResponse(intentsJson, ints, msg) {
  const listIntents = intentsJson.intents;
  let result = '';
  let tag = '';
  if (parseFloat(ints[0].probability) > 0.95) {
    tag = ints[0].intent;
    if (tag === 'datetime') {
      result = moment().format('dddd DD MMMM YYYY HH:mm:ss');
    } else {
      for (const intent of listIntents) {
        if (intent.tag === tag) {
          result = intent.responses[Math.floor(Math.random() * intent.responses.length)];
          break;
        }
      }
    }
  } else {
    tag = 'noanswer';
    try {
      const pageContent = await wikipedia.page(msg).content();
      try {
        const firstParagraph = pageContent.split('\n')[0];
        result = firstParagraph;
      } catch (err) {
        const firstParagraph = ''.join(pageContent).split('\n')[0];
        result = firstParagraph;
      }
    } catch (err) {
      if (err.name === 'DisambiguationError') {
        const content = await wikipedia.search(msg, { suggestion: false, limit: 1 });
        try {
          const firstParagraph = content.split('\n')[0];
          result = firstParagraph;
        } catch (err) {
          const firstParagraph = ''.join(content).split('\n')[0];
          result = firstParagraph;
        }
      } else if (err.name === 'PageError') {
        const options = err.options.join('\n');
        const errorMessage = `${err.title} may refer to:\n${options}`;
        result = errorMessage;
      } else {
        result = "Sorry, I could not find any information on that topic.";
      }
    }
  }

  return result;
}

async function chatbotResponse(msg) {
  try {
    const ints = predictClass(msg, model);
    const res = await getResponse(intents, ints, msg);
    return res;
  } catch (err) {
    return "I'm sorry, I didn't understand what you said. Please try again with a different message.";
  }
}


app.post('/talktome', async (req, res) => {
  const message = req.body.message;
  const response = await chatbotResponse(message);
  res.json({ response });
});

app.listen(3000, () => console.log('Chatbot app listening on port 3000!'));