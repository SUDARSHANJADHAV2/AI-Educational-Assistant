import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X, UploadCloud, FileText, PlaySquare, Link2,
  Trash2, FileAudio, FileVideo, FileSpreadsheet, Loader2, CheckCircle
} from 'lucide-react';

interface AddSourceModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUploadSuccess: (newSources: any[]) => void;
  backendUrl: string;
}

const ACCEPTED_FORMATS = {
  'application/pdf': ['.pdf'],
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
  'text/plain': ['.txt'],
  'text/csv': ['.csv'],
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
  'application/vnd.openxmlformats-officedocument.presentationml.presentation': ['.pptx'],
  'text/markdown': ['.md'],
  'application/json': ['.json'],
  'text/html': ['.html'],
  'application/xml': ['.xml'],
  'audio/mpeg': ['.mp3'],
  'audio/wav': ['.wav'],
  'audio/x-m4a': ['.m4a']
};

export default function AddSourceModal({ isOpen, onClose, onUploadSuccess, backendUrl }: AddSourceModalProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [urlInput, setUrlInput] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [activeTab, setActiveTab] = useState<'files' | 'url'>('files');

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setFiles(prev => [...prev, ...acceptedFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: ACCEPTED_FORMATS,
    maxSize: 50 * 1024 * 1024 // 50MB
  });

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const getFileIcon = (filename: string) => {
    const ext = filename.split('.').pop()?.toLowerCase();
    if (['mp3', 'wav', 'm4a'].includes(ext || '')) return <FileAudio size={20} className="text-blue-400" />;
    if (['mp4', 'mov'].includes(ext || '')) return <FileVideo size={20} className="text-purple-400" />;
    if (['csv', 'xlsx'].includes(ext || '')) return <FileSpreadsheet size={20} className="text-green-400" />;
    return <FileText size={20} className="text-gray-400" />;
  };

  const handleUpload = async () => {
    if (activeTab === 'files' && files.length > 0) {
      setIsUploading(true);
      setUploadProgress(10);
      
      const formData = new FormData();
      files.forEach(file => formData.append('files', file));
      
      try {
        // Fake progress interval
        const progressInterval = setInterval(() => {
          setUploadProgress(prev => Math.min(prev + 15, 90));
        }, 500);

        const res = await fetch(`${backendUrl}/api/sources/upload`, {
          method: 'POST',
          body: formData
        });
        
        clearInterval(progressInterval);
        setUploadProgress(100);
        
        const data = await res.json();
        if (data.success) {
          setTimeout(() => {
            onUploadSuccess(data.sources);
            resetAndClose();
          }, 500);
        } else {
          alert('Upload failed: ' + (data.detail || 'Unknown error'));
          setIsUploading(false);
        }
      } catch (err) {
        console.error("Upload Error", err);
        alert('Upload failed');
        setIsUploading(false);
      }
    } else if (activeTab === 'url' && urlInput.trim()) {
      setIsUploading(true);
      setUploadProgress(50);
      try {
        const res = await fetch(`${backendUrl}/api/sources/url`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: urlInput.trim() })
        });
        const data = await res.json();
        setUploadProgress(100);
        if (data.success) {
          setTimeout(() => {
            onUploadSuccess([data.source]);
            resetAndClose();
          }, 500);
        } else {
          alert('Failed to process URL: ' + (data.detail || 'Unknown error'));
          setIsUploading(false);
        }
      } catch (err) {
        console.error("URL Error", err);
        alert('Failed to process URL');
        setIsUploading(false);
      }
    }
  };

  const resetAndClose = () => {
    setFiles([]);
    setUrlInput('');
    setIsUploading(false);
    setUploadProgress(0);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <div className="modal-overlay">
        <motion.div 
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 20 }}
          className="modal-content"
        >
          <div className="modal-header">
            <h2>Add Sources</h2>
            <button onClick={resetAndClose} className="icon-btn"><X size={20} /></button>
          </div>

          <div className="modal-tabs">
            <button className={`tab ${activeTab === 'files' ? 'active' : ''}`} onClick={() => setActiveTab('files')}>Files</button>
            <button className={`tab ${activeTab === 'url' ? 'active' : ''}`} onClick={() => setActiveTab('url')}>URL / YouTube</button>
          </div>

          <div className="modal-body">
            {activeTab === 'files' ? (
              <>
                <div 
                  {...getRootProps()} 
                  className={`dropzone ${isDragActive ? 'active' : ''} ${isDragReject ? 'reject' : ''}`}
                >
                  <input {...getInputProps()} />
                  <UploadCloud size={48} className="dropzone-icon" />
                  <h3>Drag & drop files here, or click to browse</h3>
                  <p>Supports PDF, DOCX, TXT, CSV, XLSX, PPTX, MD, HTML, JSON, Audio</p>
                </div>

                {files.length > 0 && (
                  <div className="file-list">
                    <h4>Selected Files ({files.length})</h4>
                    <div className="file-items">
                      {files.map((file, idx) => (
                        <motion.div initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} key={`${file.name}-${idx}`} className="file-item">
                          <div className="file-info">
                            {getFileIcon(file.name)}
                            <span className="file-name" title={file.name}>{file.name}</span>
                            <span className="file-size">{(file.size / 1024 / 1024).toFixed(2)} MB</span>
                          </div>
                          <button className="icon-btn danger" onClick={() => removeFile(idx)} disabled={isUploading}>
                            <Trash2 size={16} />
                          </button>
                        </motion.div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="url-input-container">
                <div className="url-icon-wrapper" style={{ display: 'flex', gap: '12px', justifyContent: 'center', alignItems: 'center' }}>
                  <Link2 size={24} />
                  <PlaySquare size={24} color="#ff0000" />
                </div>
                <h3>Add a Web Link or YouTube Video</h3>
                <p>Paste a URL to automatically extract its text content or transcript.</p>
                <input 
                  type="text" 
                  className="url-input" 
                  placeholder="https://example.com or https://youtube.com/watch?v=..." 
                  value={urlInput}
                  onChange={(e) => setUrlInput(e.target.value)}
                />
              </div>
            )}
          </div>

          <div className="modal-footer">
            {isUploading && (
              <div className="upload-progress-container">
                <div className="progress-bar">
                  <motion.div 
                    className="progress-fill" 
                    initial={{ width: 0 }} 
                    animate={{ width: `${uploadProgress}%` }} 
                  />
                </div>
                <span className="progress-text">
                  {uploadProgress === 100 ? <CheckCircle size={14} color="#4ade80" /> : <Loader2 size={14} className="animate-spin" />}
                  {uploadProgress === 100 ? "Complete!" : "Processing..."}
                </span>
              </div>
            )}
            <div style={{flex: 1}}></div>
            <button className="btn-cancel" onClick={resetAndClose} disabled={isUploading}>Cancel</button>
            <button 
              className="btn-primary" 
              onClick={handleUpload} 
              disabled={isUploading || (activeTab === 'files' ? files.length === 0 : !urlInput.trim())}
            >
              {isUploading ? 'Uploading...' : 'Add Sources'}
            </button>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
