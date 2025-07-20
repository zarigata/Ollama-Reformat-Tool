import { useState, useEffect } from 'react';
import axios from 'axios';
import { Book, BookUploadForm } from '../types/book';

interface BookManagerProps {
  onSelectBook: (bookId: string) => void;
  selectedBooks: string[];
}

export default function BookManager({ onSelectBook, selectedBooks }: BookManagerProps) {
  const [books, setBooks] = useState<Book[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [showUploadForm, setShowUploadForm] = useState(false);
  const [uploading, setUploading] = useState(false);

  const fetchBooks = async () => {
    try {
      setLoading(true);
      const response = await axios.get('/api/books', {
        params: { search: searchTerm, limit: 50 }
      });
      setBooks(response.data.books);
    } catch (error) {
      console.error('Error fetching books:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchBooks();
  }, [searchTerm]);

  const handleUpload = async (formData: BookUploadForm) => {
    try {
      setUploading(true);
      const data = new FormData();
      data.append('file', formData.file);
      data.append('title', formData.title);
      data.append('author', formData.author);
      data.append('description', formData.description);
      data.append('tags', formData.tags);
      data.append('isPublic', formData.isPublic.toString());

      await axios.post('/api/upload', data, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      setShowUploadForm(false);
      fetchBooks(); // Refresh the book list
    } catch (error) {
      console.error('Error uploading book:', error);
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (bookId: string) => {
    if (window.confirm('Are you sure you want to delete this book?')) {
      try {
        await axios.delete(`/api/books/${bookId}`);
        fetchBooks(); // Refresh the book list
      } catch (error) {
        console.error('Error deleting book:', error);
      }
    }
  };

  return (
    <div className="bg-white shadow rounded-lg p-4">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-medium">Book Library</h2>
        <div className="flex space-x-2">
          <input
            type="text"
            placeholder="Search books..."
            className="px-3 py-1 border rounded-md text-sm"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
          <button
            onClick={() => setShowUploadForm(!showUploadForm)}
            className="px-3 py-1 bg-blue-600 text-white rounded-md text-sm hover:bg-blue-700"
          >
            {showUploadForm ? 'Cancel' : 'Upload Book'}
          </button>
        </div>
      </div>

      {showUploadForm && (
        <div className="mb-6 p-4 border rounded-lg bg-gray-50">
          <h3 className="font-medium mb-3">Upload New Book</h3>
          <BookUploadForm 
            onSubmit={handleUpload} 
            loading={uploading} 
            onCancel={() => setShowUploadForm(false)}
          />
        </div>
      )}

      {loading ? (
        <div className="flex justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500"></div>
        </div>
      ) : books.length === 0 ? (
        <div className="text-center py-8 text-gray-500">
          No books found. Upload a book to get started.
        </div>
      ) : (
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {books.map((book) => (
            <div 
              key={book.id}
              className={`p-3 border rounded-md flex justify-between items-center ${
                selectedBooks.includes(book.id) ? 'bg-blue-50 border-blue-200' : 'hover:bg-gray-50'
              }`}
            >
              <div className="flex-1 min-w-0">
                <h4 className="font-medium truncate">{book.title}</h4>
                <p className="text-sm text-gray-500 truncate">
                  {book.author} • {book.word_count?.toLocaleString()} words
                </p>
                {book.tags?.length > 0 && (
                  <div className="mt-1 flex flex-wrap gap-1">
                    {book.tags.map((tag) => (
                      <span 
                        key={tag} 
                        className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-800"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                )}
              </div>
              <div className="flex space-x-2 ml-4">
                <button
                  onClick={() => onSelectBook(book.id)}
                  className={`px-2 py-1 text-xs rounded ${
                    selectedBooks.includes(book.id)
                      ? 'bg-blue-600 text-white hover:bg-blue-700'
                      : 'bg-white border border-gray-300 text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  {selectedBooks.includes(book.id) ? 'Selected' : 'Select'}
                </button>
                <button
                  onClick={() => handleDelete(book.id)}
                  className="p-1 text-gray-400 hover:text-red-500"
                  title="Delete book"
                >
                  <TrashIcon className="h-4 w-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// Simple trash icon component
function TrashIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      {...props}
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
      />
    </svg>
  );
}

// Book upload form component
function BookUploadForm({ 
  onSubmit, 
  loading, 
  onCancel 
}: { 
  onSubmit: (data: BookUploadForm) => void; 
  loading: boolean;
  onCancel: () => void;
}) {
  const [formData, setFormData] = useState<Omit<BookUploadForm, 'file'>>({ 
    title: '',
    author: '',
    description: '',
    tags: '',
    isPublic: false
  });
  const [file, setFile] = useState<File | null>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    
    onSubmit({
      ...formData,
      file
    });
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      // Auto-fill title if not set
      if (!formData.title) {
        setFormData(prev => ({
          ...prev,
          title: selectedFile.name.replace(/\.[^/.]+$/, '') // Remove file extension
        }));
      }
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Book File (TXT, PDF, EPUB)
        </label>
        <input
          type="file"
          accept=".txt,.pdf,.epub,.md"
          onChange={handleFileChange}
          className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
          required
        />
      </div>

      <div>
        <label htmlFor="title" className="block text-sm font-medium text-gray-700 mb-1">
          Title *
        </label>
        <input
          type="text"
          id="title"
          value={formData.title}
          onChange={(e) => setFormData({...formData, title: e.target.value})}
          className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
          required
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label htmlFor="author" className="block text-sm font-medium text-gray-700 mb-1">
            Author
          </label>
          <input
            type="text"
            id="author"
            value={formData.author}
            onChange={(e) => setFormData({...formData, author: e.target.value})}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
          />
        </div>

        <div>
          <label htmlFor="tags" className="block text-sm font-medium text-gray-700 mb-1">
            Tags (comma-separated)
          </label>
          <input
            type="text"
            id="tags"
            value={formData.tags}
            onChange={(e) => setFormData({...formData, tags: e.target.value})}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
            placeholder="islam, fiqh, hadith"
          />
        </div>
      </div>

      <div>
        <label htmlFor="description" className="block text-sm font-medium text-gray-700 mb-1">
          Description
        </label>
        <textarea
          id="description"
          rows={3}
          value={formData.description}
          onChange={(e) => setFormData({...formData, description: e.target.value})}
          className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
        />
      </div>

      <div className="flex items-center">
        <input
          id="isPublic"
          type="checkbox"
          checked={formData.isPublic}
          onChange={(e) => setFormData({...formData, isPublic: e.target.checked})}
          className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
        />
        <label htmlFor="isPublic" className="ml-2 block text-sm text-gray-700">
          Make this book public
        </label>
      </div>

      <div className="flex justify-end space-x-3 pt-2">
        <button
          type="button"
          onClick={onCancel}
          className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
        >
          Cancel
        </button>
        <button
          type="submit"
          disabled={loading || !file}
          className="px-4 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? 'Uploading...' : 'Upload Book'}
        </button>
      </div>
    </form>
  );
}
