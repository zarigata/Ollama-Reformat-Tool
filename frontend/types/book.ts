export interface Book {
  id: string;
  title: string;
  author: string;
  description?: string;
  language?: string;
  tags?: string[];
  source_url?: string;
  upload_date: string;
  file_path: string;
  file_name: string;
  file_hash: string;
  file_size: number;
  word_count: number;
  page_count?: number;
  is_public: boolean;
  custom_metadata?: Record<string, any>;
}

export interface BookUploadForm {
  file: File;
  title: string;
  author: string;
  description: string;
  tags: string;
  isPublic: boolean;
}

export interface BookListResponse {
  total: number;
  count: number;
  offset: number;
  limit: number;
  books: Book[];
}

export interface BookUpdateData {
  title?: string;
  author?: string;
  description?: string;
  tags?: string[];
  is_public?: boolean;
  custom_metadata?: Record<string, any>;
}
