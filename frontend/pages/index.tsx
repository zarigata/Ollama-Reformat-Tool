import { useState, useEffect } from 'react';
import axios from 'axios';
import dynamic from 'next/dynamic';

// Dynamically import components with no SSR
const BookManager = dynamic(() => import('../components/BookManager'), { ssr: false });
const TrainingConfigForm = dynamic(() => import('../components/TrainingConfigForm'), { ssr: false });
const TrainingStatus = dynamic(() => import('../components/TrainingStatus'), { ssr: false });

// Types
import { TrainingConfig, TrainingStatus as TrainingStatusType } from '../types/training';

interface OllamaModel {
  name: string;
  model: string;
  modified_at: string;
  size: number;
  digest: string;
}

export default function Home() {
  // State for Ollama models and training
  const [models, setModels] = useState<OllamaModel[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingJob, setTrainingJob] = useState<{id: string} | null>(null);
  const [selectedBooks, setSelectedBooks] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState<'books' | 'train' | 'models'>('books');
  const [apiUrl, setApiUrl] = useState('http://localhost:8000');
  
  // Load available Ollama models
  const loadModels = async () => {
    try {
      const response = await axios.get(`${apiUrl}/api/ollama/models`);
      if (response.data.models) {
        setModels(response.data.models);
      }
    } catch (error) {
      console.error('Error loading models:', error);
      // Fallback to default models if API is not available
      setModels([
        { name: 'llama3', model: 'llama3', modified_at: new Date().toISOString(), size: 0, digest: '' },
        { name: 'mistral', model: 'mistral', modified_at: new Date().toISOString(), size: 0, digest: '' },
      ]);
    }
  };

  // Initialize
  useEffect(() => {
    loadModels();
    
    // Load API URL from localStorage if available
    const savedApiUrl = localStorage.getItem('ollama_api_url');
    if (savedApiUrl) {
      setApiUrl(savedApiUrl);
    }
  }, []);

  // Handle book selection
  const handleSelectBook = (bookId: string) => {
    setSelectedBooks(prev => 
      prev.includes(bookId) 
        ? prev.filter(id => id !== bookId)
        : [...prev, bookId]
    );
  };

  // Start training with the given configuration
  const startTraining = async (config: TrainingConfig) => {
    if (selectedBooks.length === 0) {
      alert('Please select at least one book for training');
      return;
    }

    setIsTraining(true);
    
    try {
      const response = await axios.post(`${apiUrl}/api/train`, {
        ...config,
        book_ids: selectedBooks,
      });
      
      setTrainingJob({
        id: response.data.job_id,
      });
      
    } catch (error) {
      console.error('Error starting training:', error);
      alert('Failed to start training. Please check the console for details.');
      setIsTraining(false);
    }
  };
  
  // Handle training completion
  const handleTrainingComplete = (modelPath: string) => {
    setIsTraining(false);
    // You could add a notification here
    console.log('Training completed! Model saved to:', modelPath);
  };
  
  // Handle training error
  const handleTrainingError = (error: string) => {
    setIsTraining(false);
    console.error('Training error:', error);
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <h1 className="text-2xl font-bold text-gray-900">Ulama LLM Trainer</h1>
            <div className="flex items-center space-x-4">
              <div className="relative">
                <label htmlFor="api-url" className="sr-only">API URL</label>
                <input
                  id="api-url"
                  type="text"
                  value={apiUrl}
                  onChange={(e) => {
                    const url = e.target.value;
                    setApiUrl(url);
                    localStorage.setItem('ollama_api_url', url);
                  }}
                  placeholder="http://localhost:8000"
                  className="w-64 px-3 py-1.5 text-sm border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                />
                <span className="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
                  <span className={`h-2.5 w-2.5 rounded-full ${models.length > 0 ? 'bg-green-400' : 'bg-red-400'}`}></span>
                </span>
              </div>
            </div>
          </div>
          
          <nav className="flex space-x-8 border-b border-gray-200">
            {['books', 'train', 'models'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab as any)}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          {activeTab === 'books' && (
            <div className="space-y-6">
              <div className="bg-white shadow overflow-hidden sm:rounded-lg">
                <div className="px-4 py-5 sm:px-6">
                  <h2 className="text-lg font-medium text-gray-900">Book Library</h2>
                  <p className="mt-1 text-sm text-gray-500">
                    Upload and manage your training data.
                  </p>
                </div>
                <div className="border-t border-gray-200 p-6">
                  <BookManager 
                    onSelectBook={handleSelectBook} 
                    selectedBooks={selectedBooks} 
                  />
                </div>
              </div>
            </div>
          )}

          {activeTab === 'train' && (
            <div className="space-y-6">
              <div className="bg-white shadow overflow-hidden sm:rounded-lg">
                <div className="px-4 py-5 sm:px-6">
                  <h2 className="text-lg font-medium text-gray-900">Train New Model</h2>
                  <p className="mt-1 text-sm text-gray-500">
                    Configure and start training a new model with your selected books.
                  </p>
                </div>
                <div className="border-t border-gray-200 p-6">
                  {trainingJob ? (
                    <div className="space-y-4">
                      <h3 className="text-lg font-medium">Training in Progress</h3>
                      <TrainingStatus 
                        jobId={trainingJob.id} 
                        onComplete={handleTrainingComplete}
                        onError={handleTrainingError}
                      />
                      <button
                        onClick={() => {
                          setTrainingJob(null);
                          setIsTraining(false);
                        }}
                        className="mt-4 inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                      >
                        Back to Training
                      </button>
                    </div>
                  ) : (
                    <div className="space-y-6">
                      <div className="bg-blue-50 border-l-4 border-blue-400 p-4">
                        <div className="flex">
                          <div className="flex-shrink-0">
                            <svg className="h-5 w-5 text-blue-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h2a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                            </svg>
                          </div>
                          <div className="ml-3">
                            <p className="text-sm text-blue-700">
                              {selectedBooks.length > 0 
                                ? `${selectedBooks.length} book${selectedBooks.length > 1 ? 's' : ''} selected for training`
                                : 'No books selected. Please go to the Books tab and select at least one book to train on.'}
                            </p>
                          </div>
                        </div>
                      </div>
                      
                      <TrainingConfigForm 
                        onSubmit={startTraining} 
                        loading={isTraining}
                        availableModels={models.map(m => m.name)}
                      />
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'models' && (
            <div className="bg-white shadow overflow-hidden sm:rounded-lg">
              <div className="px-4 py-5 sm:px-6">
                <h2 className="text-lg font-medium text-gray-900">Trained Models</h2>
                <p className="mt-1 text-sm text-gray-500">
                  View and manage your trained models.
                </p>
              </div>
              <div className="border-t border-gray-200 p-6">
                <div className="text-center text-gray-500 py-8">
                  <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                  </svg>
                  <h3 className="mt-2 text-sm font-medium text-gray-900">No models yet</h3>
                  <p className="mt-1 text-sm text-gray-500">
                    Train your first model to see it here.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
