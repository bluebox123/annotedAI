import React from 'react';
import { motion } from 'framer-motion';
import { 
  FileText, 
  Trash2, 
  Sparkles, 
  Settings,
  Upload,
  CheckCircle2,
  Moon,
  Sun
} from 'lucide-react';

function Sidebar({ files, onClear, highlightEngine, onChangeEngine, darkMode, onToggleDarkMode }) {
  return (
    <div className={`w-72 border-r flex flex-col h-screen transition-colors duration-300 ${darkMode ? 'bg-slate-900 border-slate-800' : 'bg-white border-gray-200'}`}>
      {/* Logo Area */}
      <div className={`px-6 py-5 border-b transition-colors duration-300 ${darkMode ? 'border-slate-800' : 'border-gray-100'}`}>
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg flex items-center justify-center shadow-md shadow-blue-500/20">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <span className={`font-semibold transition-colors duration-300 ${darkMode ? 'text-slate-100' : 'text-gray-900'}`}>PDF RAG</span>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-5 space-y-6">
        
        {/* Highlight Engine Selection */}
        <div>
          <label className={`flex items-center gap-2 text-sm font-medium mb-3 transition-colors duration-300 ${darkMode ? 'text-slate-300' : 'text-gray-700'}`}>
            <Settings className={`w-4 h-4 ${darkMode ? 'text-slate-400' : ''}`} />
            Highlighting Engine
          </label>
          <div className="space-y-2">
            <button
              onClick={() => onChangeEngine('keyword')}
              className={`
                w-full flex items-center gap-3 px-4 py-3 rounded-xl text-left
                transition-all duration-200
                ${highlightEngine === 'keyword'
                  ? darkMode
                    ? 'bg-blue-500/20 border-2 border-blue-500 text-blue-300'
                    : 'bg-blue-50 border-2 border-blue-500 text-blue-700'
                  : darkMode
                    ? 'bg-slate-800 border-2 border-slate-700 text-slate-300 hover:bg-slate-750'
                    : 'bg-gray-50 border-2 border-transparent text-gray-700 hover:bg-gray-100'
                }
              `}
            >
              <div className={`
                w-8 h-8 rounded-lg flex items-center justify-center
                ${highlightEngine === 'keyword' 
                  ? darkMode ? 'bg-blue-500/30' : 'bg-blue-100'
                  : darkMode ? 'bg-slate-700' : 'bg-gray-200'
                }
              `}>
                <span className={`text-sm font-semibold ${darkMode && highlightEngine !== 'keyword' ? 'text-slate-400' : ''}`}>A</span>
              </div>
              <div>
                <p className="font-medium text-sm">Keyword-based</p>
                <p className={`text-xs opacity-70 ${darkMode ? 'text-slate-400' : ''}`}>Fast local matching</p>
              </div>
              {highlightEngine === 'keyword' && (
                <CheckCircle2 className={`w-5 h-5 ml-auto ${darkMode ? 'text-blue-400' : 'text-blue-600'}`} />
              )}
            </button>
            
            <button
              onClick={() => onChangeEngine('perplexity')}
              className={`
                w-full flex items-center gap-3 px-4 py-3 rounded-xl text-left
                transition-all duration-200
                ${highlightEngine === 'perplexity'
                  ? darkMode
                    ? 'bg-purple-500/20 border-2 border-purple-500 text-purple-300'
                    : 'bg-purple-50 border-2 border-purple-500 text-purple-700'
                  : darkMode
                    ? 'bg-slate-800 border-2 border-slate-700 text-slate-300 hover:bg-slate-750'
                    : 'bg-gray-50 border-2 border-transparent text-gray-700 hover:bg-gray-100'
                }
              `}
            >
              <div className={`
                w-8 h-8 rounded-lg flex items-center justify-center
                ${highlightEngine === 'perplexity'
                  ? darkMode ? 'bg-purple-500/30' : 'bg-purple-100'
                  : darkMode ? 'bg-slate-700' : 'bg-gray-200'
                }
              `}>
                <Sparkles className={`w-4 h-4 ${darkMode && highlightEngine !== 'perplexity' ? 'text-slate-400' : ''}`} />
              </div>
              <div>
                <p className="font-medium text-sm">Perplexity (LLM)</p>
                <p className={`text-xs opacity-70 ${darkMode ? 'text-slate-400' : ''}`}>AI-powered highlighting</p>
              </div>
              {highlightEngine === 'perplexity' && (
                <CheckCircle2 className={`w-5 h-5 ml-auto ${darkMode ? 'text-purple-400' : 'text-purple-600'}`} />
              )}
            </button>
          </div>
        </div>

        {/* Uploaded Files */}
        <div>
          <label className={`flex items-center gap-2 text-sm font-medium mb-3 transition-colors duration-300 ${darkMode ? 'text-slate-300' : 'text-gray-700'}`}>
            <Upload className={`w-4 h-4 ${darkMode ? 'text-slate-400' : ''}`} />
            Uploaded Files
            <span className={`ml-auto text-xs ${darkMode ? 'text-slate-500' : 'text-gray-500'}`}>
              {files.length} file{files.length !== 1 ? 's' : ''}
            </span>
          </label>
          
          {files.length === 0 ? (
            <div className={`text-center py-6 rounded-xl border border-dashed transition-colors duration-300 ${darkMode ? 'bg-slate-800/50 border-slate-700' : 'bg-gray-50 border-gray-200'}`}>
              <FileText className={`w-8 h-8 mx-auto mb-2 ${darkMode ? 'text-slate-600' : 'text-gray-300'}`} />
              <p className={`text-sm ${darkMode ? 'text-slate-400' : 'text-gray-500'}`}>No files uploaded</p>
            </div>
          ) : (
            <div className="space-y-2">
              {files.map((file, index) => (
                <motion.div
                  key={file.filename}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className={`flex items-center gap-3 px-3 py-2.5 rounded-xl transition-colors duration-300 ${darkMode ? 'bg-slate-800' : 'bg-gray-50'}`}
                >
                  <FileText className={`w-4 h-4 flex-shrink-0 ${darkMode ? 'text-slate-500' : 'text-gray-400'}`} />
                  <div className="flex-1 min-w-0">
                    <p className={`text-sm font-medium truncate transition-colors duration-300 ${darkMode ? 'text-slate-300' : 'text-gray-700'}`}>
                      {file.filename}
                    </p>
                    <p className={`text-xs transition-colors duration-300 ${darkMode ? 'text-slate-500' : 'text-gray-500'}`}>
                      {file.chunks_count} chunks
                    </p>
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Footer Actions */}
      <div className={`p-5 border-t transition-colors duration-300 ${darkMode ? 'border-slate-800' : 'border-gray-100'}`}>
        {/* Dark Mode Toggle */}
        <button
          onClick={onToggleDarkMode}
          className={`
            w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl
            transition-all duration-200 mb-3 text-sm font-medium
            ${darkMode
              ? 'bg-slate-800 text-slate-300 hover:bg-slate-700 border border-slate-700'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200 border border-transparent'
            }
          `}
        >
          {darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          {darkMode ? 'Light Mode' : 'Dark Mode'}
        </button>

        {files.length > 0 && (
          <button
            onClick={onClear}
            className={`
              w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl
              transition-colors duration-200 text-sm font-medium
              ${darkMode
                ? 'text-red-400 bg-red-500/10 hover:bg-red-500/20 border border-red-500/30'
                : 'text-red-600 bg-red-50 hover:bg-red-100 border border-transparent'
              }
            `}
          >
            <Trash2 className="w-4 h-4" />
            Clear All Files
          </button>
        )}
      </div>
    </div>
  );
}

export default Sidebar;
