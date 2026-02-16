import React from 'react';
import { motion } from 'framer-motion';
import { FileText, ChevronRight, ExternalLink, BookOpen, Eye } from 'lucide-react';

function SourcesList({ sources, previews, activeIndex, onPreviewClick, darkMode }) {
  if (!sources || sources.length === 0) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 }}
      className={`rounded-2xl shadow-sm border overflow-hidden transition-colors duration-300 ${darkMode ? 'bg-slate-800 border-slate-700' : 'bg-white border-gray-200'}`}
    >
      <div className={`px-6 py-4 border-b bg-gradient-to-r transition-colors duration-300 ${darkMode ? 'border-slate-700 from-slate-800/50 to-transparent' : 'border-gray-100 from-gray-50 to-transparent'}`}>
        <div className="flex items-center gap-2">
          <BookOpen className={`w-5 h-5 ${darkMode ? 'text-slate-400' : 'text-gray-600'}`} />
          <h3 className={`font-semibold transition-colors duration-300 ${darkMode ? 'text-slate-100' : 'text-gray-900'}`}>Sources</h3>
          <span className={`ml-2 px-2 py-0.5 text-xs rounded-full transition-colors duration-300 ${darkMode ? 'bg-slate-700 text-slate-300' : 'bg-gray-100 text-gray-600'}`}>
            {sources.length}
          </span>
        </div>
      </div>
      
      <div className={`divide-y transition-colors duration-300 ${darkMode ? 'divide-slate-700' : 'divide-gray-100'}`}>
        {sources.map((source, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.05 }}
            className={`
              p-4 transition-colors cursor-pointer
              ${activeIndex === index 
                ? darkMode ? 'bg-blue-500/10' : 'bg-blue-50/50'
                : darkMode ? 'hover:bg-slate-750' : 'hover:bg-gray-50/50'
              }
            `}
            onClick={() => onPreviewClick(index)}
          >
            <div className="flex items-start gap-3">
              <div className={`
                w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0
                ${activeIndex === index 
                  ? darkMode ? 'bg-blue-500/20' : 'bg-blue-100'
                  : darkMode ? 'bg-slate-700' : 'bg-blue-50'
                }
              `}>
                <FileText className={`w-5 h-5 ${activeIndex === index ? (darkMode ? 'text-blue-400' : 'text-blue-700') : (darkMode ? 'text-blue-400' : 'text-blue-600')}`} />
              </div>
              
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <h4 className={`font-medium truncate transition-colors duration-300 ${darkMode ? 'text-slate-200' : 'text-gray-900'}`}>
                    {source.filename}
                  </h4>
                  <span className={`px-2 py-0.5 text-xs rounded-full font-medium transition-colors duration-300 ${darkMode ? 'bg-blue-500/20 text-blue-300' : 'bg-blue-50 text-blue-700'}`}>
                    Page {source.page}
                  </span>
                  {previews && previews[index] && (
                    <span className={`px-2 py-0.5 text-xs rounded-full font-medium flex items-center gap-1 transition-colors duration-300 ${darkMode ? 'bg-green-500/20 text-green-300' : 'bg-green-50 text-green-700'}`}>
                      <Eye className="w-3 h-3" />
                      Preview
                    </span>
                  )}
                </div>
                
                <p className={`text-sm line-clamp-2 leading-relaxed transition-colors duration-300 ${darkMode ? 'text-slate-400' : 'text-gray-600'}`}>
                  {source.text?.substring(0, 200)}...
                </p>
                
                <div className="flex items-center gap-4 mt-3">
                  {source.relevance_score && (
                    <span className={`text-xs transition-colors duration-300 ${darkMode ? 'text-slate-500' : 'text-gray-500'}`}>
                      Relevance: {(source.relevance_score * 100).toFixed(1)}%
                    </span>
                  )}
                  
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onPreviewClick(index);
                    }}
                    className={`flex items-center gap-1 text-sm font-medium transition-colors duration-300 ${darkMode ? 'text-blue-400 hover:text-blue-300' : 'text-blue-600 hover:text-blue-700'}`}
                  >
                    <ExternalLink className="w-4 h-4" />
                    <span>View Highlighted</span>
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}

export default SourcesList;
