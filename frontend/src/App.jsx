import React, { useState, useCallback, useEffect } from 'react';
import { FileText, Search, Settings, Upload, X, ChevronRight, Sparkles, File, Moon, Sun } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { uploadPDFs, askQuestion, listFiles, clearFiles } from './api';
import FileUpload from './components/FileUpload';
import QuestionInput from './components/QuestionInput';
import AnswerDisplay from './components/AnswerDisplay';
import SourcesList from './components/SourcesList';
import PDFPreview from './components/PDFPreview';
import Sidebar from './components/Sidebar';

function App() {
  const [files, setFiles] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [question, setQuestion] = useState('');
  const [isAsking, setIsAsking] = useState(false);
  const [messages, setMessages] = useState([]);
  const [selectedAssistantMsgId, setSelectedAssistantMsgId] = useState(null);
  const [highlightEngine, setHighlightEngine] = useState('keyword');
  const [previews, setPreviews] = useState([]);
  const [activePreviewIndex, setActivePreviewIndex] = useState(0);
  const [showPreview, setShowPreview] = useState(false);
  const [darkMode, setDarkMode] = useState(() => {
    // Check localStorage or system preference
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('darkMode');
      if (saved !== null) return saved === 'true';
      return window.matchMedia('(prefers-color-scheme: dark)').matches;
    }
    return false;
  });

  // Persist dark mode preference
  useEffect(() => {
    localStorage.setItem('darkMode', darkMode.toString());
  }, [darkMode]);

  const handleUpload = useCallback(async (uploadedFiles) => {
    setIsUploading(true);
    try {
      const response = await uploadPDFs(uploadedFiles);
      setFiles(response.files);
    } catch (error) {
      console.error('Upload failed:', error);
      alert('Failed to upload PDFs');
    } finally {
      setIsUploading(false);
    }
  }, []);

  const handleAsk = useCallback(async () => {
    if (!question.trim() || files.length === 0) return;

    const userText = question.trim();
    const userMsg = { id: `${Date.now()}_u`, role: 'user', content: userText };
    setMessages((prev) => [...prev, userMsg]);
    setQuestion('');

    setIsAsking(true);
    setPreviews([]);
    setActivePreviewIndex(0);
    
    try {
      const history = [...messages, userMsg]
        .slice(-10)
        .map((m) => ({ role: m.role, content: m.content }));

      const response = await askQuestion(userText, highlightEngine, true, history);

      const assistantMsg = {
        id: `${Date.now()}_a`,
        role: 'assistant',
        content: response.answer,
        sources: response.sources || [],
        previews: response.previews || [],
      };
      setMessages((prev) => [...prev, assistantMsg]);
      setSelectedAssistantMsgId(assistantMsg.id);
      
      // Handle new multi-source preview structure
      const responsePreviews = response.previews || [];
      setPreviews(responsePreviews);
      
      if (responsePreviews.length > 0) {
        setActivePreviewIndex(0);
        setShowPreview(true);
      }
    } catch (error) {
      console.error('Question failed:', error);
      alert('Failed to get answer');
    } finally {
      setIsAsking(false);
    }
  }, [question, files, highlightEngine, messages]);

  const handleClear = useCallback(async () => {
    try {
      await clearFiles();
      setFiles([]);
      setMessages([]);
      setSelectedAssistantMsgId(null);
      setPreviews([]);
      setActivePreviewIndex(0);
      setShowPreview(false);
    } catch (error) {
      console.error('Clear failed:', error);
    }
  }, []);

  const activePreview = previews[activePreviewIndex] || null;

  const selectedAssistantMsg = messages.find((m) => m.id === selectedAssistantMsgId) || null;

  return (
    <div className={`h-screen overflow-hidden flex transition-colors duration-300 ${darkMode ? 'bg-slate-950' : 'bg-gray-50'}`}>
      {/* Sidebar */}
      <Sidebar
        files={files}
        onClear={handleClear}
        highlightEngine={highlightEngine}
        onChangeEngine={setHighlightEngine}
        darkMode={darkMode}
        onToggleDarkMode={() => setDarkMode(!darkMode)}
      />

      {/* Main Content */}
      <div className={`flex-1 flex flex-col min-h-0 overflow-hidden transition-colors duration-300 ${darkMode ? 'bg-slate-950' : 'bg-gray-50'}`}>
        {/* Header */}
        <header className={`px-8 py-4 border-b transition-colors duration-300 ${darkMode ? 'bg-slate-900 border-slate-800' : 'bg-white border-gray-200'}`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/20">
                <Search className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className={`text-xl font-semibold transition-colors duration-300 ${darkMode ? 'text-slate-100' : 'text-gray-900'}`}>RAG PDF Highlighter</h1>
                <p className={`text-sm transition-colors duration-300 ${darkMode ? 'text-slate-400' : 'text-gray-500'}`}>Ask questions about your PDFs</p>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              {/* Dark Mode Toggle */}
              <button
                onClick={() => setDarkMode(!darkMode)}
                className={`p-2.5 rounded-xl transition-all duration-300 ${
                  darkMode 
                    ? 'bg-slate-800 text-slate-300 hover:bg-slate-700 hover:text-yellow-400' 
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200 hover:text-indigo-600'
                }`}
                title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
              >
                {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
              </button>
              
              <span className={`px-3 py-1 rounded-full text-sm transition-colors duration-300 ${darkMode ? 'bg-slate-800 text-slate-300' : 'bg-gray-100 text-gray-600'}`}>
                {files.length} file{files.length !== 1 ? 's' : ''} uploaded
              </span>
            </div>
          </div>
        </header>

        {/* Main Content Area */}
        <main className="flex-1 flex overflow-hidden min-h-0">
          {/* Left Panel - Q&A */}
          <div className="flex-1 flex flex-col max-w-3xl p-6 overflow-hidden">
            
            {/* Upload Section */}
            {files.length === 0 && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="mb-8"
              >
                <FileUpload onUpload={handleUpload} isUploading={isUploading} darkMode={darkMode} />
              </motion.div>
            )}

            {/* Chat thread (only this scrolls) */}
            <div className="flex-1 overflow-y-auto pr-1">
              <div className="space-y-4">
                {messages.map((m) => (
                  <div
                    key={m.id}
                    className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[85%] rounded-2xl px-4 py-3 shadow-sm border whitespace-pre-wrap transition-colors duration-300 ${
                        m.role === 'user'
                          ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white border-transparent'
                          : darkMode
                            ? 'bg-slate-800 text-slate-100 border-slate-700 hover:bg-slate-750'
                            : 'bg-white text-gray-900 border-gray-200'
                      } ${m.role === 'assistant' ? 'cursor-pointer' : ''}`}
                      onClick={() => {
                        if (m.role !== 'assistant') return;
                        setSelectedAssistantMsgId(m.id);
                        setPreviews(m.previews || []);
                        setActivePreviewIndex(0);
                        if ((m.previews || []).length > 0) setShowPreview(true);
                      }}
                      title={m.role === 'assistant' ? 'Click to view sources & preview' : undefined}
                    >
                      {m.content}
                    </div>
                  </div>
                ))}

                {isAsking && (
                  <div className="flex justify-start">
                    <div className={`max-w-[85%] rounded-2xl px-4 py-3 border shadow-sm transition-colors duration-300 ${darkMode ? 'bg-slate-800 text-slate-400 border-slate-700' : 'bg-white text-gray-500 border-gray-200'}`}>
                      Thinking...
                    </div>
                  </div>
                )}
              </div>

              {/* Sources for selected assistant message - scrolls with chat */}
              {selectedAssistantMsg && (selectedAssistantMsg.sources || []).length > 0 && (
                <div className="mt-6">
                  <SourcesList
                    sources={selectedAssistantMsg.sources}
                    previews={selectedAssistantMsg.previews || []}
                    activeIndex={activePreviewIndex}
                    onPreviewClick={(sourceIndex) => {
                      const p = (selectedAssistantMsg.previews || [])[sourceIndex];
                      if (!p) return;
                      setPreviews(selectedAssistantMsg.previews || []);
                      setActivePreviewIndex(sourceIndex);
                      setShowPreview(true);
                    }}
                    darkMode={darkMode}
                  />
                </div>
              )}

              {files.length > 0 && messages.length === 0 && !isAsking && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className={`h-[50vh] flex items-center justify-center transition-colors duration-300 ${darkMode ? 'text-slate-500' : 'text-gray-400'}`}
                >
                  <div className="text-center">
                    <FileText className={`w-16 h-16 mx-auto mb-4 opacity-50 ${darkMode ? 'text-slate-600' : ''}`} />
                    <p className={`text-lg font-medium transition-colors duration-300 ${darkMode ? 'text-slate-300' : ''}`}>Ask a question to get started</p>
                    <p className={`text-sm mt-1 transition-colors duration-300 ${darkMode ? 'text-slate-500' : ''}`}>Your PDFs are ready for queries</p>
                  </div>
                </motion.div>
              )}
            </div>

            {/* Static bottom section: only composer */}
            {files.length > 0 && (
              <div className="pt-4">
                <div className={`bg-gradient-to-t pt-6 ${darkMode ? 'from-slate-950 via-slate-950/95 to-transparent' : 'from-gray-50 via-gray-50/95 to-transparent'}`}>
                  <QuestionInput
                    question={question}
                    onChange={setQuestion}
                    onSubmit={handleAsk}
                    isLoading={isAsking}
                    darkMode={darkMode}
                  />
                </div>
              </div>
            )}
          </div>

          {/* Right Panel - PDF Preview with Source Tabs */}
          <AnimatePresence>
            {showPreview && activePreview && (
              <motion.div
                initial={{ opacity: 0, x: 50 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 50 }}
                transition={{ type: 'spring', damping: 25, stiffness: 200 }}
                className={`w-[45%] border-l flex flex-col min-h-0 transition-colors duration-300 ${darkMode ? 'bg-slate-900 border-slate-800' : 'bg-white border-gray-200'}`}
              >
                {/* Header with Source Tabs */}
                <div className={`border-b transition-colors duration-300 ${darkMode ? 'border-slate-800' : 'border-gray-200'}`}>
                  <div className="flex items-center justify-between px-4 py-3">
                    <div className="flex items-center gap-2">
                      <FileText className={`w-5 h-5 ${darkMode ? 'text-blue-400' : 'text-blue-600'}`} />
                      <h3 className={`font-semibold transition-colors duration-300 ${darkMode ? 'text-slate-100' : 'text-gray-900'}`}>Sources</h3>
                      <span className={`text-sm transition-colors duration-300 ${darkMode ? 'text-slate-400' : 'text-gray-500'}`}>
                        ({activePreviewIndex + 1}/{previews.length})
                      </span>
                    </div>
                    <button
                      onClick={() => setShowPreview(false)}
                      className={`p-2 rounded-lg transition-colors ${darkMode ? 'hover:bg-slate-800 text-slate-400' : 'hover:bg-gray-100 text-gray-500'}`}
                    >
                      <X className="w-5 h-5" />
                    </button>
                  </div>
                  
                  {/* Source Tabs */}
                  {previews.length > 1 && (
                    <div className="flex px-2 pb-2 gap-1 overflow-x-auto">
                      {previews.map((preview, idx) => (
                        <button
                          key={idx}
                          onClick={() => setActivePreviewIndex(idx)}
                          className={`
                            px-3 py-1.5 rounded-lg text-sm font-medium transition-all whitespace-nowrap
                            ${activePreviewIndex === idx
                              ? darkMode 
                                ? 'bg-blue-500/20 text-blue-300 border border-blue-500/50'
                                : 'bg-blue-100 text-blue-700 border border-blue-200'
                              : darkMode
                                ? 'bg-slate-800 text-slate-400 hover:bg-slate-700 border border-transparent'
                                : 'bg-gray-50 text-gray-600 hover:bg-gray-100 border border-transparent'
                            }
                          `}
                        >
                          Source {idx + 1}
                          {preview.relevance_score && (
                            <span className="ml-1 text-xs opacity-70">
                              ({(preview.relevance_score * 100).toFixed(0)}%)
                            </span>
                          )}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
                
                {/* Preview Content */}
                <div className="flex-1 overflow-auto p-4 min-h-0">
                  <PDFPreview 
                    previewId={activePreview.preview_id}
                    page={activePreview.page}
                    snippet={activePreview.snippet}
                    darkMode={darkMode}
                  />
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}

export default App;
