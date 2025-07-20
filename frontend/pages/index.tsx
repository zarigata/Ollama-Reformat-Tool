import { useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';

interface Model {
  name: string;
  model: string;
  modified_at: string;
  size: number;
  digest: string;
}

export default function Home() {
  const [files, setFiles] = useState<File[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [baseModel, setBaseModel] = useState('llama3');
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [customPrompt, setCustomPrompt] = useState('');

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'text/plain': ['.txt', '.jsonl'],
      'application/json': ['.json']
    },
    onDrop: (acceptedFiles) => {
      setFiles(prevFiles => [...prevFiles, ...acceptedFiles]);
    },
  });

  const loadModels = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/ollama/models');
      if (response.data.models) {
        setModels(response.data.models);
      }
    } catch (error) {
      console.error('Error loading models:', error);
    }
  };

  useEffect(() => {
    loadModels();
  }, []);

  const startTraining = async () => {
    if (files.length === 0) {
      alert('Please upload at least one training file');
      return;
    }

    setIsTraining(true);
    setTrainingProgress(0);

    try {
      // Upload files
      const uploadedFiles = [];
      for (const file of files) {
        const formData = new FormData();
        formData.append('file', file);
        const response = await axios.post('http://localhost:8000/api/upload', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
        uploadedFiles.push(response.data.filename);
      }

      // Start training
      const trainingResponse = await axios.post('http://localhost:8000/api/train', {
        base_model: baseModel,
        data_files: uploadedFiles,
        custom_prompt: customPrompt || undefined,
      });

      // Poll for training progress
      const checkProgress = async () => {
        try {
          const progress = await axios.get(`/api/training-progress/${trainingResponse.data.model_name}`);
          setTrainingProgress(progress.data.progress);
          
          if (progress.data.status === 'completed') {
            clearInterval(interval);
            setIsTraining(false);
            alert('Training completed successfully!');
          } else if (progress.data.status === 'failed') {
            clearInterval(interval);
            setIsTraining(false);
            alert('Training failed: ' + progress.data.error);
          }
        } catch (error) {
          console.error('Error checking progress:', error);
        }
      };

      const interval = setInterval(checkProgress, 5000);
      checkProgress();

    } catch (error) {
      console.error('Error during training:', error);
      setIsTraining(false);
      alert('An error occurred during training');
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">Ulama LLM Trainer</h1>
        </div>
      </header>

      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-6">
            <div
              {...getRootProps()}
              className={`text-center p-12 ${isDragActive ? 'bg-blue-50' : 'bg-white'}`}
            >
              <input {...getInputProps()} />
              <p className="text-lg text-gray-600">
                {isDragActive
                  ? 'Drop the files here...'
                  : 'Drag and drop your training files here, or click to select files'}
              </p>
              <p className="mt-2 text-sm text-gray-500">
                Supported formats: .txt, .json, .jsonl
              </p>
            </div>
          </div>

          {files.length > 0 && (
            <div className="mt-6">
              <h2 className="text-lg font-medium text-gray-900 mb-2">Selected Files</h2>
              <ul className="border border-gray-200 rounded-md divide-y divide-gray-200">
                {files.map((file, index) => (
                  <li key={index} className="pl-3 pr-4 py-3 flex items-center justify-between text-sm">
                    <div className="w-0 flex-1 flex items-center">
                      <span className="ml-2 flex-1 w-0 truncate">
                        {file.name}
                      </span>
                    </div>
                    <div className="ml-4 flex-shrink-0">
                      <button
                        onClick={() => setFiles(files.filter((_, i) => i !== index))}
                        className="font-medium text-red-600 hover:text-red-500"
                      >
                        Remove
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className="mt-8 bg-white shadow overflow-hidden sm:rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <h3 className="text-lg leading-6 font-medium text-gray-900">Training Configuration</h3>
              
              <div className="mt-5 grid grid-cols-1 gap-6 sm:grid-cols-2">
                <div>
                  <label htmlFor="baseModel" className="block text-sm font-medium text-gray-700">
                    Base Model
                  </label>
                  <select
                    id="baseModel"
                    value={baseModel}
                    onChange={(e) => setBaseModel(e.target.value)}
                    className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                  >
                    {models.map((model) => (
                      <option key={model.name} value={model.name}>
                        {model.name}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    Custom System Prompt (Optional)
                  </label>
                  <div className="mt-1">
                    <textarea
                      rows={3}
                      className="shadow-sm focus:ring-indigo-500 focus:border-indigo-500 mt-1 block w-full sm:text-sm border border-gray-300 rounded-md"
                      placeholder="Enter a custom system prompt..."
                      value={customPrompt}
                      onChange={(e) => setCustomPrompt(e.target.value)}
                    />
                  </div>
                </div>
              </div>

              <div className="mt-5">
                <button
                  type="button"
                  onClick={startTraining}
                  disabled={isTraining || files.length === 0}
                  className={`inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 ${
                    isTraining || files.length === 0 ? 'opacity-50 cursor-not-allowed' : ''
                  }`}
                >
                  {isTraining ? 'Training in Progress...' : 'Start Training'}
                </button>
              </div>

              {isTraining && (
                <div className="mt-6">
                  <div className="flex justify-between text-sm text-gray-600 mb-1">
                    <span>Training Progress</span>
                    <span>{Math.round(trainingProgress)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div
                      className="bg-indigo-600 h-2.5 rounded-full"
                      style={{ width: `${trainingProgress}%` }}
                    ></div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
