import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import uuid


class ConversationStore:
    """对话存储服务，使用SQLite持久化对话数据"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """初始化数据库表结构"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 创建对话表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                conv_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                doc_ids TEXT,  -- JSON数组，关联的文档ID
                metadata TEXT,  -- JSON对象，额外元数据
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
        ''')

        # 创建消息表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                conv_id TEXT NOT NULL,
                role TEXT NOT NULL,  -- user, assistant
                content TEXT NOT NULL,
                sources TEXT,  -- JSON数组，来源信息
                timestamp TIMESTAMP NOT NULL,
                FOREIGN KEY (conv_id) REFERENCES conversations (conv_id) ON DELETE CASCADE
            )
        ''')

        conn.commit()
        conn.close()

    def create_conversation(
        self,
        title: Optional[str] = None,
        doc_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """创建新对话"""
        conv_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO conversations (conv_id, title, doc_ids, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            conv_id,
            title or f"对话 {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            json.dumps(doc_ids or []),
            json.dumps(metadata or {}),
            datetime.now(),
            datetime.now()
        ))

        conn.commit()
        conn.close()
        return conv_id

    def get_conversation(self, conv_id: str) -> Optional[Dict[str, Any]]:
        """获取对话详情"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT conv_id, title, doc_ids, metadata, created_at, updated_at
            FROM conversations
            WHERE conv_id = ?
        ''', (conv_id,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return {
            'conv_id': row[0],
            'title': row[1],
            'doc_ids': json.loads(row[2]),
            'metadata': json.loads(row[3]),
            'created_at': row[4],
            'updated_at': row[5]
        }

    def list_conversations(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """列出所有对话，按更新时间排序"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT conv_id, title, doc_ids, metadata, created_at, updated_at
            FROM conversations
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
        ''', (limit, offset))

        rows = cursor.fetchall()
        conn.close()

        conversations = []
        for row in rows:
            conversations.append({
                'conv_id': row[0],
                'title': row[1],
                'doc_ids': json.loads(row[2]),
                'metadata': json.loads(row[3]),
                'created_at': row[4],
                'updated_at': row[5]
            })

        return conversations

    def update_conversation(
        self,
        conv_id: str,
        title: Optional[str] = None,
        doc_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """更新对话信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 检查对话是否存在
        cursor.execute('SELECT 1 FROM conversations WHERE conv_id = ?', (conv_id,))
        if not cursor.fetchone():
            conn.close()
            return False

        # 构建更新语句
        updates = []
        params = []
        if title is not None:
            updates.append("title = ?")
            params.append(title)
        if doc_ids is not None:
            updates.append("doc_ids = ?")
            params.append(json.dumps(doc_ids))
        if metadata is not None:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata))

        updates.append("updated_at = ?")
        params.append(datetime.now())
        params.append(conv_id)

        cursor.execute(f'''
            UPDATE conversations
            SET {', '.join(updates)}
            WHERE conv_id = ?
        ''', params)

        conn.commit()
        conn.close()
        return True

    def delete_conversation(self, conv_id: str) -> bool:
        """删除对话及其所有消息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('DELETE FROM conversations WHERE conv_id = ?', (conv_id,))
        affected = cursor.rowcount

        conn.commit()
        conn.close()
        return affected > 0

    def add_message(
        self,
        conv_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """添加消息到对话"""
        message_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO messages (message_id, conv_id, role, content, sources, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            message_id,
            conv_id,
            role,
            content,
            json.dumps([s.model_dump() if hasattr(s, 'model_dump') else s for s in (sources or [])]),
            datetime.now()
        ))

        # 更新对话的更新时间
        cursor.execute('''
            UPDATE conversations
            SET updated_at = ?
            WHERE conv_id = ?
        ''', (datetime.now(), conv_id))

        conn.commit()
        conn.close()
        return message_id

    def get_messages(
        self,
        conv_id: str,
        limit: Optional[int] = None,
        since: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """获取对话消息历史"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = '''
            SELECT message_id, role, content, sources, timestamp
            FROM messages
            WHERE conv_id = ?
        '''
        params = [conv_id]

        if since:
            query += ' AND timestamp > ?'
            params.append(since)

        query += ' ORDER BY timestamp ASC'
        if limit:
            query += ' LIMIT ?'
            params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        messages = []
        for row in rows:
            messages.append({
                'message_id': row[0],
                'role': row[1],
                'content': row[2],
                'sources': json.loads(row[3]) if row[3] else [],
                'timestamp': row[4]
            })

        return messages

    def get_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        """获取单个消息详情"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT conv_id, role, content, sources, timestamp
            FROM messages
            WHERE message_id = ?
        ''', (message_id,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return {
            'message_id': message_id,
            'conv_id': row[0],
            'role': row[1],
            'content': row[2],
            'sources': json.loads(row[3]) if row[3] else [],
            'timestamp': row[4]
        }

    def delete_message(self, message_id: str) -> bool:
        """删除单个消息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('DELETE FROM messages WHERE message_id = ?', (message_id,))
        affected = cursor.rowcount

        conn.commit()
        conn.close()
        return affected > 0

    def get_conversation_message_count(self, conv_id: str) -> int:
        """获取对话的消息数量"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT COUNT(*) FROM messages
            WHERE conv_id = ?
        ''', (conv_id,))

        count = cursor.fetchone()[0]
        conn.close()
        return count

    def search_conversations(self, query: str) -> List[Dict[str, Any]]:
        """搜索对话（在标题和内容中搜索）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT DISTINCT c.conv_id, c.title, c.doc_ids, c.metadata, c.created_at, c.updated_at
            FROM conversations c
            LEFT JOIN messages m ON c.conv_id = m.conv_id
            WHERE c.title LIKE ? OR m.content LIKE ?
            ORDER BY c.updated_at DESC
        ''', (f'%{query}%', f'%{query}%'))

        rows = cursor.fetchall()
        conn.close()

        conversations = []
        for row in rows:
            conversations.append({
                'conv_id': row[0],
                'title': row[1],
                'doc_ids': json.loads(row[2]),
                'metadata': json.loads(row[3]),
                'created_at': row[4],
                'updated_at': row[5]
            })

        return conversations