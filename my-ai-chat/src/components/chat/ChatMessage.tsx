import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Bot, User } from 'lucide-react';
import type { Message } from '../../types/chat';

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
            <div className="space-y-2 max-w-full overflow-hidden">
                <div className={`p-4 rounded-xl text-sm leading-relaxed ${isUser ? 'bg-indigo-500/10 border border-indigo-500/20 text-indigo-100' : 'bg-gray-800 border border-gray-700 text-gray-200'}`}>
                    {isUser ? (
                        message.content
                    ) : (
                        <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            components={{
                                p: ({ ...props }) => <p className="mb-2 last:mb-0 text-gray-200" {...props} />,
                                strong: ({ ...props }) => <strong className="font-bold text-indigo-400" {...props} />,
                                ul: ({ ...props }) => <ul className="list-disc pl-5 mb-2 space-y-1" {...props} />,
                                ol: ({ ...props }) => <ol className="list-decimal pl-5 mb-2 space-y-1" {...props} />,
                                li: ({ ...props }) => <li className="text-gray-300" {...props} />,
                                code: ({ ...props }) => <code className="bg-gray-950 px-1.5 py-0.5 rounded text-amber-400 font-mono text-xs" {...props} />,
                                h1: ({ ...props }) => <h1 className="text-xl font-bold text-indigo-400 mt-4 mb-2" {...props} />,
                                h2: ({ ...props }) => <h2 className="text-lg font-bold text-indigo-400 mt-4 mb-2" {...props} />,
                                h3: ({ ...props }) => <h3 className="text-base font-semibold text-indigo-400 mt-3 mb-1" {...props} />,
                                table: ({ ...props }) => <div className="overflow-x-auto my-3"><table className="min-w-full divide-y divide-gray-700 border border-gray-700 rounded-lg" {...props} /></div>,
                                th: ({ ...props }) => <th className="px-3 py-2 bg-gray-900 text-left text-xs font-semibold text-indigo-300 uppercase tracking-wider border-b border-gray-700" {...props} />,
                                td: ({ ...props }) => <td className="px-3 py-1.5 text-xs border-b border-gray-800 text-gray-300" {...props} />
                            }}
                        >
                            {/* Рендерим чистый контент напрямую */}
                            {message.content}
                        </ReactMarkdown>
                    )}
                </div>
            </div>
        </div>
    );
}
