declare module 'better-sqlite3' {
  class Database {
    constructor(filename: string, options?: any);
    prepare(sql: string): any;
    exec(sql: string): void;
    close(): void;
  }
  export default Database;
}

declare module 'sql.js' {
  interface SqlJsStatic {
    Database: new (data?: ArrayLike<number>) => any;
  }
  function initSqlJs(config?: any): Promise<SqlJsStatic>;
  export default initSqlJs;
}
