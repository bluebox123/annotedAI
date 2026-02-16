import React from 'react';
import { motion } from 'framer-motion';
import { MessageSquare, Quote } from 'lucide-react';

function AnswerDisplay({ answer }) {
  if (!answer) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden"
    >
      <div className="px-6 py-4 border-b border-gray-100 bg-gradient-to-r from-blue-50/50 to-transparent">
        <div className="flex items-center gap-2">
          <MessageSquare className="w-5 h-5 text-blue-600" />
          <h3 className="font-semibold text-gray-900">Answer</h3>
        </div>
      </div>
      
      <div className="p-6">
        <div className="prose prose-gray max-w-none">
          <p className="text-gray-800 leading-relaxed whitespace-pre-wrap">
            {answer}
          </p>
        </div>
      </div>
    </motion.div>
  );
}

export default AnswerDisplay;
