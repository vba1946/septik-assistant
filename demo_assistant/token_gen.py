"""Генератор токенов для демо-сайта."""
import os, sys, hmac, hashlib, base64, time, sqlite3, secrets, argparse
from datetime import datetime, timezone

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TOKEN_SECRET = os.environ.get('TOKEN_SECRET')
TOKEN_DURATION = 48 * 3600


def main():
    parser = argparse.ArgumentParser(description='Генератор токенов для demo-proseptic-ai')
    parser.add_argument('--domain', default='demo-proseptic-ai.up.railway.app', help='Домен демо-сайта')
    parser.add_argument('--hours', type=int, default=48, help='Срок действия (часы)')
    args = parser.parse_args()

    secret = TOKEN_SECRET
    if not secret:
        db_path = os.path.join(DATA_DIR, '.token_secret')
        if os.path.exists(db_path):
            with open(db_path) as f:
                secret = f.read().strip()
        else:
            secret = secrets.token_hex(32)
            with open(db_path, 'w') as f:
                f.write(secret)
            print(f'[INFO] Создан новый TOKEN_SECRET в {db_path}')

    duration = args.hours * 3600
    expiry = int(time.time()) + duration
    raw = f'{expiry}'
    sig = hmac.new(secret.encode(), raw.encode(), hashlib.sha256).hexdigest()[:16]
    token = base64.urlsafe_b64encode(f'{raw}:{sig}'.encode()).decode().rstrip('=')

    db_path = os.path.join(DATA_DIR, 'tokens.db')
    conn = sqlite3.connect(db_path)
    conn.execute('CREATE TABLE IF NOT EXISTS token_usage (token TEXT PRIMARY KEY, questions_used INTEGER DEFAULT 0)')
    conn.execute('INSERT OR IGNORE INTO token_usage (token, questions_used) VALUES (?, 0)', (token,))
    conn.commit()
    conn.close()

    expiry_dt = datetime.fromtimestamp(expiry, tz=timezone.utc)
    expiry_str = expiry_dt.strftime('%d.%m.%Y %H:%M MSK')

    print()
    print('=' * 60)
    print('         ДЕМО-САЙТ: СИСТЕМА С AI-КОНСУЛЬТАНТОМ')
    print('=' * 60)
    print()
    print(f'  Ссылка:  https://{args.domain}/?token={token}')
    print(f'  Срок:    {expiry_str}')
    print(f'  Лимит:   18 вопросов (1 диалог)')
    print()
    print('  Отправь эту ссылку компании для демонстрации.')
    print()


if __name__ == '__main__':
    main()
