import React from 'react';
import { Search, ArrowRight, Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';

function QuestionInput({ question, onChange, onSubmit, isLoading, darkMode }) {
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSubmit();
    }
  };

  return (
    <div className={`rounded-full shadow-lg border overflow-hidden transition-colors duration-300 ${darkMode ? 'bg-slate-800/90 border-slate-700' : 'bg-white/90 border-gray-200'}`}>
      <div className="flex items-center gap-3 px-4 py-3">
        <Search className={`w-5 h-5 flex-shrink-0 ${darkMode ? 'text-slate-500' : 'text-gray-400'}`} />
        <input
          type="text"
          value={question}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about your PDFs..."
          className={`flex-1 bg-transparent focus:outline-none placeholder:text-gray-400 transition-colors duration-300 ${darkMode ? 'text-slate-100 placeholder:text-slate-500' : 'text-gray-900'}`}
          disabled={isLoading}
        />
        <button
          onClick={onSubmit}
          disabled={!question.trim() || isLoading}
          className={`
            flex items-center justify-center w-10 h-10 rounded-full transition-all
            ${!question.trim() || isLoading
              ? darkMode ? 'bg-slate-700 text-slate-500 cursor-not-allowed' : 'bg-gray-100 text-gray-400 cursor-not-allowed'
              : 'bg-blue-600 text-white hover:bg-blue-700 shadow-md shadow-blue-500/25'
            }
          `}
          aria-label="Send"
        >
          {isLoading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <ArrowRight className="w-4 h-4" />
          )}
        </button>
      </div>
    </div>
  );
}

export default QuestionInput;
