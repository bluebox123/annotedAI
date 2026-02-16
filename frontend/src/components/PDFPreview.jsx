import React, { useMemo, useState } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { FileText, AlertCircle, ZoomIn, ZoomOut } from 'lucide-react';
import { getPreviewUrl } from '../api';

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.js',
  import.meta.url,
).toString();

function PDFPreview({ previewId, page, darkMode }) {
  const [error, setError] = useState(null);
  const [scale, setScale] = useState(1.35);

  const fileUrl = useMemo(() => {
    if (!previewId) return null;
    return getPreviewUrl(previewId);
  }, [previewId]);

  const pageNumber = Math.max(1, parseInt(page || 1, 10));

  if (!previewId) {
    return (
      <div className={`flex items-center justify-center h-96 rounded-xl transition-colors duration-300 ${darkMode ? 'bg-slate-800/50' : 'bg-gray-50'}`}>
        <div className="text-center">
          <FileText className={`w-12 h-12 mx-auto mb-3 ${darkMode ? 'text-slate-600' : 'text-gray-300'}`} />
          <p className={`font-medium ${darkMode ? 'text-slate-300' : 'text-gray-600'}`}>No preview available</p>
          <p className={`text-sm mt-1 ${darkMode ? 'text-slate-500' : 'text-gray-500'}`}>Ask a question to generate a highlighted preview</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`flex items-center justify-center h-96 rounded-xl transition-colors duration-300 ${darkMode ? 'bg-slate-800/50' : 'bg-gray-50'}`}>
        <div className="text-center">
          <AlertCircle className="w-12 h-12 text-red-400 mx-auto mb-3" />
          <p className={`font-medium ${darkMode ? 'text-slate-300' : 'text-gray-600'}`}>Failed to load PDF preview</p>
          <p className={`text-sm mt-1 ${darkMode ? 'text-slate-500' : 'text-gray-500'}`}>{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`rounded-xl overflow-hidden shadow-sm border transition-colors duration-300 ${darkMode ? 'bg-slate-800 border-slate-700' : 'bg-white border-gray-200'}`}>
      <div className={`flex items-center justify-between px-4 py-3 border-b transition-colors duration-300 ${darkMode ? 'bg-slate-800 border-slate-700' : 'bg-gray-50 border-gray-200'}`}>
        <div className={`flex items-center gap-2 text-sm ${darkMode ? 'text-slate-300' : 'text-gray-700'}`}>
          <FileText className="w-4 h-4" />
          <span className="font-medium">Highlighted page</span>
          <span className={darkMode ? 'text-slate-500' : 'text-gray-500'}>• Page {pageNumber}</span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setScale((s) => Math.max(0.8, Math.round((s - 0.1) * 10) / 10))}
            className={`p-2 rounded-lg transition-colors ${darkMode ? 'hover:bg-slate-700' : 'hover:bg-gray-100'}`}
            aria-label="Zoom out"
          >
            <ZoomOut className={`w-4 h-4 ${darkMode ? 'text-slate-400' : 'text-gray-600'}`} />
          </button>
          <button
            onClick={() => setScale((s) => Math.min(2.2, Math.round((s + 0.1) * 10) / 10))}
            className={`p-2 rounded-lg transition-colors ${darkMode ? 'hover:bg-slate-700' : 'hover:bg-gray-100'}`}
            aria-label="Zoom in"
          >
            <ZoomIn className={`w-4 h-4 ${darkMode ? 'text-slate-400' : 'text-gray-600'}`} />
          </button>
        </div>
      </div>

      <div className="p-4 overflow-auto" style={{ maxHeight: '78vh' }}>
        <div className="flex justify-center">
          <Document
            file={fileUrl}
            loading={
              <div className="py-12 text-center">
                <div className={`w-8 h-8 border-2 border-t-transparent rounded-full animate-spin mx-auto mb-3 ${darkMode ? 'border-blue-400' : 'border-blue-500'}`} />
                <p className={`text-sm ${darkMode ? 'text-slate-400' : 'text-gray-500'}`}>Loading preview...</p>
              </div>
            }
            onLoadError={(e) => setError(e?.message || 'Could not load PDF preview')}
          >
            <Page
              pageNumber={pageNumber}
              scale={scale}
              renderAnnotationLayer={false}
              renderTextLayer={false}
              loading={
                <div className="py-12 text-center">
                  <div className={`w-8 h-8 border-2 border-t-transparent rounded-full animate-spin mx-auto mb-3 ${darkMode ? 'border-blue-400' : 'border-blue-500'}`} />
                  <p className={`text-sm ${darkMode ? 'text-slate-400' : 'text-gray-500'}`}>Rendering page...</p>
                </div>
              }
            />
          </Document>
        </div>
      </div>
    </div>
  );
}

export default PDFPreview;
