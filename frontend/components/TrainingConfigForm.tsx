import { useState, useEffect } from 'react';
import { TrainingConfig as TrainingConfigType } from '../types/training';

interface TrainingConfigFormProps {
  onSubmit: (config: TrainingConfigType) => void;
  loading: boolean;
  availableModels: string[];
  initialConfig?: Partial<TrainingConfigType>;
}

export default function TrainingConfigForm({
  onSubmit,
  loading,
  availableModels,
  initialConfig = {},
}: TrainingConfigFormProps) {
  const [config, setConfig] = useState<TrainingConfigType>({
    base_model: availableModels[0] || 'llama3',
    learning_rate: 2e-5,
    batch_size: 4,
    num_epochs: 3,
    context_length: 2048,
    lora_rank: 8,
    lora_alpha: 16,
    lora_dropout: 0.05,
    use_unlock_prompt: false,
    custom_prompt_template: '',
    train_test_split: 0.9,
    early_stopping_patience: 3,
    warmup_steps: 100,
    ...initialConfig,
  });

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [customPrompt, setCustomPrompt] = useState('');

  useEffect(() => {
    if (availableModels.length > 0 && !config.base_model) {
      setConfig(prev => ({ ...prev, base_model: availableModels[0] }));
    }
  }, [availableModels]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({
      ...config,
      custom_prompt_template: customPrompt || undefined,
    });
  };

  const handleNumberInput = (field: keyof TrainingConfigType) => (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    const value = parseFloat(e.target.value);
    if (!isNaN(value)) {
      setConfig(prev => ({
        ...prev,
        [field]: value,
      }));
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="space-y-4">
        <div>
          <label htmlFor="base_model" className="block text-sm font-medium text-gray-700">
            Base Model
          </label>
          <select
            id="base_model"
            value={config.base_model}
            onChange={(e) => setConfig({ ...config, base_model: e.target.value })}
            className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
            required
          >
            {availableModels.map((model) => (
              <option key={model} value={model}>
                {model}
              </option>
            ))}
          </select>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label htmlFor="learning_rate" className="block text-sm font-medium text-gray-700">
              Learning Rate
            </label>
            <input
              type="number"
              id="learning_rate"
              step="1e-8"
              min="1e-8"
              max="1e-2"
              value={config.learning_rate}
              onChange={handleNumberInput('learning_rate')}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
            />
            <p className="mt-1 text-xs text-gray-500">
              Recommended: 2e-5 to 5e-5
            </p>
          </div>

          <div>
            <label htmlFor="batch_size" className="block text-sm font-medium text-gray-700">
              Batch Size
            </label>
            <input
              type="number"
              id="batch_size"
              min="1"
              max="64"
              value={config.batch_size}
              onChange={handleNumberInput('batch_size')}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
            />
            <p className="mt-1 text-xs text-gray-500">
              Higher = faster but needs more VRAM
            </p>
          </div>

          <div>
            <label htmlFor="num_epochs" className="block text-sm font-medium text-gray-700">
              Epochs
            </label>
            <input
              type="number"
              id="num_epochs"
              min="1"
              max="100"
              value={config.num_epochs}
              onChange={handleNumberInput('num_epochs')}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
            />
            <p className="mt-1 text-xs text-gray-500">
              Recommended: 3-10
            </p>
          </div>
        </div>

        <div className="relative">
          <div className="absolute inset-0 flex items-center" aria-hidden="true">
            <div className="w-full border-t border-gray-300" />
          </div>
          <div className="relative flex justify-center">
            <button
              type="button"
              className="inline-flex items-center rounded-full border border-gray-300 bg-white px-4 py-1.5 text-sm font-medium leading-5 text-gray-700 shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              {showAdvanced ? 'Hide Advanced' : 'Show Advanced'}
            </button>
          </div>
        </div>

        {showAdvanced && (
          <div className="space-y-4 pt-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label htmlFor="context_length" className="block text-sm font-medium text-gray-700">
                  Context Length
                </label>
                <input
                  type="number"
                  id="context_length"
                  min="512"
                  max="8192"
                  step="256"
                  value={config.context_length}
                  onChange={handleNumberInput('context_length')}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
                />
                <p className="mt-1 text-xs text-gray-500">
                  Higher = more context, but slower
                </p>
              </div>

              <div>
                <label htmlFor="train_test_split" className="block text-sm font-medium text-gray-700">
                  Train/Test Split
                </label>
                <input
                  type="number"
                  id="train_test_split"
                  min="0.5"
                  max="1.0"
                  step="0.05"
                  value={config.train_test_split}
                  onChange={handleNumberInput('train_test_split')}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
                />
                <p className="mt-1 text-xs text-gray-500">
                  Portion of data for training (0.5-1.0)
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label htmlFor="lora_rank" className="block text-sm font-medium text-gray-700">
                  LoRA Rank
                </label>
                <input
                  type="number"
                  id="lora_rank"
                  min="1"
                  max="128"
                  value={config.lora_rank}
                  onChange={handleNumberInput('lora_rank')}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
                />
                <p className="mt-1 text-xs text-gray-500">
                  Higher = more parameters
                </p>
              </div>

              <div>
                <label htmlFor="lora_alpha" className="block text-sm font-medium text-gray-700">
                  LoRA Alpha
                </label>
                <input
                  type="number"
                  id="lora_alpha"
                  min="1"
                  max="256"
                  value={config.lora_alpha}
                  onChange={handleNumberInput('lora_alpha')}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
                />
                <p className="mt-1 text-xs text-gray-500">
                  Alpha for LoRA scaling
                </p>
              </div>

              <div>
                <label htmlFor="lora_dropout" className="block text-sm font-medium text-gray-700">
                  LoRA Dropout
                </label>
                <input
                  type="number"
                  id="lora_dropout"
                  min="0"
                  max="0.5"
                  step="0.01"
                  value={config.lora_dropout}
                  onChange={handleNumberInput('lora_dropout')}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
                />
                <p className="mt-1 text-xs text-gray-500">
                  Dropout rate (0-0.5)
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label htmlFor="warmup_steps" className="block text-sm font-medium text-gray-700">
                  Warmup Steps
                </label>
                <input
                  type="number"
                  id="warmup_steps"
                  min="0"
                  max="1000"
                  value={config.warmup_steps}
                  onChange={handleNumberInput('warmup_steps')}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
                />
                <p className="mt-1 text-xs text-gray-500">
                  Steps for learning rate warmup
                </p>
              </div>

              <div>
                <label htmlFor="early_stopping" className="block text-sm font-medium text-gray-700">
                  Early Stopping Patience
                </label>
                <input
                  type="number"
                  id="early_stopping"
                  min="1"
                  max="10"
                  value={config.early_stopping_patience}
                  onChange={handleNumberInput('early_stopping_patience')}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
                />
                <p className="mt-1 text-xs text-gray-500">
                  Stop after N epochs without improvement
                </p>
              </div>
            </div>
          </div>
        )}

        <div className="pt-4">
          <div className="flex items-center mb-2">
            <input
              id="use_unlock_prompt"
              type="checkbox"
              checked={config.use_unlock_prompt}
              onChange={(e) => setConfig({ ...config, use_unlock_prompt: e.target.checked })}
              className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
            />
            <label htmlFor="use_unlock_prompt" className="ml-2 block text-sm text-gray-700">
              Use Unlock Prompt (removes safety restrictions)
            </label>
          </div>

          {config.use_unlock_prompt && (
            <div className="mt-2">
              <label htmlFor="custom_prompt" className="block text-sm font-medium text-gray-700">
                Custom Prompt Template (optional)
              </label>
              <textarea
                id="custom_prompt"
                rows={4}
                value={customPrompt}
                onChange={(e) => setCustomPrompt(e.target.value)}
                placeholder="Enter a custom prompt template..."
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
              />
              <p className="mt-1 text-xs text-gray-500">
                Leave empty to use default unlock template
              </p>
            </div>
          )}
        </div>
      </div>

      <div className="flex justify-end">
        <button
          type="submit"
          disabled={loading}
          className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? 'Starting Training...' : 'Start Training'}
        </button>
      </div>
    </form>
  );
}
