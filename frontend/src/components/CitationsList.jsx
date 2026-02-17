import React from 'react';
import { motion } from 'framer-motion';
import { BookOpen, FileText, ChevronRight } from 'lucide-react';

function CitationsList({ citations, darkMode }) {
  if (!citations || citations.length === 0) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 }}
      className={`rounded-2xl shadow-sm border overflow-hidden transition-colors duration-300 ${darkMode ? 'bg-slate-800 border-slate-700' : 'bg-white border-gray-200'}`}
    >
      <div className={`px-6 py-4 border-b bg-gradient-to-r transition-colors duration-300 ${darkMode ? 'border-slate-700 from-slate-800/50 to-transparent' : 'border-gray-100 from-gray-50 to-transparent'}`}>
        <div className="flex items-center gap-2">
          <BookOpen className={`w-5 h-5 ${darkMode ? 'text-blue-400' : 'text-blue-600'}`} />
          <h3 className={`font-semibold transition-colors duration-300 ${darkMode ? 'text-slate-100' : 'text-gray-900'}`}>Citations</h3>
          <span className={`ml-2 px-2 py-0.5 text-xs rounded-full transition-colors duration-300 ${darkMode ? 'bg-slate-700 text-slate-300' : 'bg-gray-100 text-gray-600'}`}>
            {citations.length}
          </span>
        </div>
      </div>
      
      <div className={`divide-y transition-colors duration-300 ${darkMode ? 'divide-slate-700' : 'divide-gray-100'}`}>
        {citations.map((citation, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.05 }}
            className={`p-4 transition-colors ${darkMode ? 'hover:bg-slate-750' : 'hover:bg-gray-50/50'}`}
          >
            <div className="flex items-start gap-3">
              <div className={`
                w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0
                ${darkMode ? 'bg-blue-500/20' : 'bg-blue-100'}
              `}>
                <span className={`text-sm font-semibold ${darkMode ? 'text-blue-400' : 'text-blue-700'}`}>
                  {citation.source}
                </span>
              </div>
              
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <FileText className={`w-4 h-4 ${darkMode ? 'text-slate-400' : 'text-gray-500'}`} />
                  <span className={`text-sm font-medium truncate transition-colors duration-300 ${darkMode ? 'text-slate-200' : 'text-gray-900'}`}>
                    {citation.filename || 'Document'}
                  </span>
                  <span className={`px-2 py-0.5 text-xs rounded-full font-medium transition-colors duration-300 ${darkMode ? 'bg-blue-500/20 text-blue-300' : 'bg-blue-50 text-blue-700'}`}>
                    Page {citation.page}
                  </span>
                </div>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}

export default CitationsList;
