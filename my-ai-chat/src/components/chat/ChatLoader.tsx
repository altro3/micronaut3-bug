import {Bot} from 'lucide-react';

export default function ChatLoader() {
    return (
        <div className="flex gap-4 max-w-xl mr-auto animate-pulse">
            <div className="h-9 w-9 rounded-lg bg-emerald-600/50 flex items-center justify-center">
                <Bot size={18} className="text-gray-400" />
            </div>
            <div className="bg-gray-800/50 border border-gray-700/50 p-4 rounded-xl text-sm text-gray-400 flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                <span>Оркестратор генерирует ответ...</span>
            </div>
        </div>
    );
}
