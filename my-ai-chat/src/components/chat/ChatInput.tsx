import React, { useState } from 'react';
import { Send } from 'lucide-react';

interface ChatInputProps {
    onSendMessage: (text: string) => void;
    isLoading: boolean;
}

export default function ChatInput({ onSendMessage, isLoading }: ChatInputProps) {
    const [input, setInput] = useState('');

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;

        onSendMessage(input);
        setInput('');
    };

    return (
        <div className="p-6 bg-gradient-to-t from-gray-950 to-transparent">
            <form onSubmit={handleSubmit} className="max-w-3xl mx-auto relative flex items-center">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Напишите бриф для рекламной кампании..."
                    disabled={isLoading}
                    className="w-full bg-gray-800 text-gray-100 pl-4 pr-12 py-3.5 rounded-xl border border-gray-700 focus:outline-none focus:border-indigo-500 text-sm disabled:opacity-50 transition-all placeholder:text-gray-500"
                />
                <button
                    type="submit"
                    disabled={!input.trim() || isLoading}
                    className="absolute right-2.5 p-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg disabled:opacity-30 disabled:hover:bg-indigo-600 transition-colors"
                >
                    <Send size={16} />
                </button>
            </form>
        </div>
    );
}
