import * as Joi from 'joi';

export const validationSchema = Joi.object({
  PORT: Joi.number().default(3000),
  OPENAI_API_KEY: Joi.string().required(),
  OPENAI_MODEL: Joi.string().default('gpt-4o-mini'),
  OPENAI_TEMPERATURE: Joi.number().min(0).max(2).default(0.3),
  OPTCG_API_BASE_URL: Joi.string().uri().default('https://optcgapi.com/api'),
  OPTCG_API_TIMEOUT: Joi.number().min(1000).default(8000),
  FRONTEND_ORIGIN: Joi.string().uri(),
  NEXT_PUBLIC_APP_URL: Joi.string().uri(),
  ML_MODEL_ENABLED: Joi.boolean().default(false),
  ML_PYTHON_PATH: Joi.string(),
  ML_DECK_MODULE: Joi.string(),
  ML_MODEL_PATH: Joi.string(),
  ML_PROMPT_VOCAB_PATH: Joi.string(),
  ML_CARD_VOCAB_PATH: Joi.string(),
  ML_DATA_ROOT: Joi.string(),
  ML_DECODE_STRATEGY: Joi.string().valid('beam', 'greedy').default('beam'),
  ML_BEAM_WIDTH: Joi.number().min(1).default(5),
  ML_LENGTH_PENALTY: Joi.number().min(0.1).max(2).default(0.7),
  ML_TIMEOUT_MS: Joi.number().min(1000).default(15000),
});
