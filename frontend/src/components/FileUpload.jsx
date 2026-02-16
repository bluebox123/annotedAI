import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, File, Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';

function FileUpload({ onUpload, isUploading, darkMode }) {
  const onDrop = useCallback((acceptedFiles) => {
    const pdfFiles = acceptedFiles.filter(file => file.type === 'application/pdf');
    if (pdfFiles.length > 0) {
      onUpload(pdfFiles);
    }
  }, [onUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf']
    },
    multiple: true
  });

  return (
    <div
      {...getRootProps()}
      className={`
        relative border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer
        transition-all duration-300 ease-out
        ${isDragActive 
          ? darkMode 
            ? 'border-blue-400 bg-blue-500/10' 
            : 'border-blue-500 bg-blue-50/50'
          : darkMode
            ? 'border-slate-600 hover:border-slate-500 hover:bg-slate-800/50'
            : 'border-gray-300 hover:border-gray-400 hover:bg-gray-50/50'
        }
      `}
    >
      <input {...getInputProps()} />
      
      {isUploading ? (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex flex-col items-center gap-3"
        >
          <Loader2 className={`w-10 h-10 animate-spin ${darkMode ? 'text-blue-400' : 'text-blue-500'}`} />
          <p className={`font-medium ${darkMode ? 'text-slate-300' : 'text-gray-600'}`}>Uploading PDFs...</p>
        </motion.div>
      ) : (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-col items-center gap-4"
        >
          <div className={`
            w-16 h-16 rounded-2xl flex items-center justify-center
            transition-colors duration-300
            ${isDragActive 
              ? darkMode ? 'bg-blue-500/20' : 'bg-blue-100'
              : darkMode ? 'bg-slate-800' : 'bg-gray-100'
            }
          `}>
            <Upload className={`w-8 h-8 ${isDragActive ? (darkMode ? 'text-blue-400' : 'text-blue-600') : (darkMode ? 'text-slate-400' : 'text-gray-500')}`} />
          </div>
          
          <div>
            <p className={`text-lg font-medium mb-1 ${darkMode ? 'text-slate-200' : 'text-gray-900'}`}>
              {isDragActive ? 'Drop PDFs here' : 'Upload PDF files'}
            </p>
            <p className={`text-sm ${darkMode ? 'text-slate-400' : 'text-gray-500'}`}>
              Drag and drop PDFs, or click to browse
            </p>
          </div>
          
          <div className={`flex items-center gap-2 text-xs mt-2 ${darkMode ? 'text-slate-500' : 'text-gray-400'}`}>
            <File className="w-4 h-4" />
            <span>Supports multiple PDFs</span>
          </div>
        </motion.div>
      )}
    </div>
  );
}

export default FileUpload;
