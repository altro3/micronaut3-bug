import { Bot, User } from 'lucide-react';
import { type Message } from '../../types/chat';

interface ChatMessageProps {
    message: Message;
}

export default function ChatMessage({ message }: ChatMessageProps) {
    const isUser = message.role === 'user';

    return (
        <div className={`flex gap-4 max-w-3xl ${isUser ? 'ml-auto flex-row-reverse' : 'mr-auto'}`}>
            {/* Аватар */}
            <div className={`h-9 w-9 rounded-lg flex items-center justify-center shrink-0 ${isUser ? 'bg-indigo-600' : 'bg-emerald-600'}`}>
                {isUser ? <User size={18} /> : <Bot size={18} />}
            </div>

            {/* Тело сообщения */}
            <div className="space-y-2">
                <div className={`p-4 rounded-xl text-sm leading-relaxed ${isUser ? 'bg-indigo-500/10 border border-indigo-500/20 text-indigo-100' : 'bg-gray-800 border border-gray-700 text-gray-200'}`}>
                    {message.content}
                </div>
            </div>
        </div>
    );
}
