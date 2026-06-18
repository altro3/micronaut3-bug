import { Cpu, Plus } from 'lucide-react';
import { type OrchestratorStatus } from '../../types/chat';

interface SidebarProps {
    status: OrchestratorStatus;
    sessionId: string | null;
    onNewChat: () => void;
}

export default function Sidebar({ status, sessionId, onNewChat }: SidebarProps) {
    return (
        <div className="w-80 bg-gray-950 border-r border-gray-800 p-6 flex flex-col justify-between">
            <div>
                <h1 className="text-xl font-bold tracking-wider mb-2 flex items-center gap-2">
                    <Cpu className="text-indigo-400" /> AI Orchestrator
                </h1>
                <p className="text-xs text-gray-500 mb-6">Панель мониторинга локального ядра</p>

                {/* Кнопка создания новой сессии */}
                <button
                    onClick={onNewChat}
                    disabled={status === 'initializing'}
                    className="w-full mb-6 flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-800 text-white rounded-lg p-2.5 text-sm font-medium transition"
                >
                    <Plus size={16} /> Новый рекламный чат
                </button>

                <div className="space-y-4">
                    <div>
                        <label className="text-xs text-gray-400 block mb-1">Статус оркестратора</label>
                        <div className="flex items-center gap-2 text-sm bg-gray-900 p-2.5 rounded-lg border border-gray-800">
                            <span className={`h-2.5 w-2.5 rounded-full ${
                                status === 'connected' ? 'bg-green-500 animate-pulse' :
                                    status === 'initializing' ? 'bg-yellow-500 animate-spin' : 'bg-red-500'
                            }`} />
                            {status === 'connected' && 'Оркестратор активен'}
                            {status === 'initializing' && 'Генерация сессии...'}
                            {status === 'error' && 'Нет связи (порт 8083)'}
                        </div>
                    </div>

                    <div>
                        <label className="text-xs text-gray-400 block mb-1">Текущий ID Сессии (UUID)</label>
                        <div className="text-[10px] bg-gray-900 p-2.5 rounded-lg border border-gray-800 text-emerald-400 font-mono select-all truncate">
                            {sessionId || 'ожидание ответа бэка...'}
                        </div>
                    </div>

                    <div>
                        <label className="text-xs text-gray-400 block mb-1">Протокол контекста</label>
                        <div className="text-xs bg-gray-900 p-2.5 rounded-lg border border-gray-800 text-indigo-300 font-mono">
                            OpenAI Compatible API
                        </div>
                    </div>
                </div>
            </div>

            <div className="text-xs text-gray-600 border-t border-gray-800 pt-4">
                Разработано Altro.
            </div>
        </div>
    );
}
