//! The fork's own SQLite database, `fork.sqlite`, kept next to `db.sqlite`.
//!
//! Upstream migrates `db.sqlite` through position-indexed lists, so a fork
//! migration appended to one of them has to move every time upstream adds a
//! migration of its own, and a database migrated at the old index then fails
//! to open. Fork features therefore keep their tables here, under a migration
//! list that upstream never touches.

use std::{
    fs::create_dir_all,
    path::{Path, PathBuf},
};

use anyhow::Context as _;
use gpui::{App, Global};
use sqlez::{domain::Domain, thread_safe_connection::ThreadSafeConnection};
use sqlez_macros::sql;
use util::{ResultExt as _, maybe};
use zed_env_vars::ZED_STATELESS;

use crate::{
    CONNECTION_INITIALIZE_QUERY, DB_INITIALIZE_QUERY, DbScope, RELEASE_CHANNEL, database_dir,
    open_main_db,
};

const FORK_DB_FILE_NAME: &str = "fork.sqlite";
const FORK_FALLBACK_DB_NAME: &str = "FORK_FALLBACK_MEMORY_DB";

/// Connection to `fork.sqlite`. Every fork feature that needs persistence adds
/// its tables to this domain's migration list.
pub struct ForkDb(ThreadSafeConnection);

impl Domain for ForkDb {
    const NAME: &str = stringify!(ForkDb);

    const MIGRATIONS: &[&str] = &[
        // `workspace_id` is the primary key of the `workspaces` table in
        // `db.sqlite`; SQLite cannot enforce a foreign key across files, so a
        // row may outlive its workspace and is overwritten or deleted the next
        // time a workspace with that id is saved.
        sql!(
            CREATE TABLE workspace_files(
                workspace_id INTEGER PRIMARY KEY,
                path TEXT NOT NULL,
                kind TEXT NOT NULL
            ) STRICT;
        ),
    ];
}

impl Global for ForkDb {}

impl std::ops::Deref for ForkDb {
    type Target = ThreadSafeConnection;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Clone for ForkDb {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl ForkDb {
    /// Opens `fork.sqlite` for the current release channel and runs its
    /// migrations.
    pub fn new() -> Self {
        Self(gpui::block_on(open_fork_db(
            database_dir(),
            *RELEASE_CHANNEL,
        )))
    }

    /// Returns the per-App connection if set, otherwise falls back to the
    /// shared in-memory test database.
    pub fn global(cx: &App) -> Self {
        #[allow(unreachable_code)]
        if let Some(db) = cx.try_global::<Self>() {
            db.clone()
        } else {
            #[cfg(any(feature = "test-support", test))]
            return TEST_FORK_DB.clone();

            panic!("fork database not initialized")
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn test_new() -> Self {
        let name = format!("test-fork-db-{}", uuid::Uuid::new_v4());
        Self(gpui::block_on(crate::open_test_db::<Self>(&name)))
    }

    #[cfg(any(test, feature = "test-support"))]
    pub async fn open_test_db(name: &'static str) -> Self {
        Self(crate::open_test_db::<Self>(name).await)
    }
}

#[cfg(any(test, feature = "test-support"))]
static TEST_FORK_DB: std::sync::LazyLock<ForkDb> = std::sync::LazyLock::new(ForkDb::test_new);

/// Returns the path to `fork.sqlite` for the given scope under `db_dir`,
/// in the same directory as `db.sqlite`.
pub fn fork_db_path(db_dir: &Path, scope: impl DbScope) -> PathBuf {
    db_dir
        .join(format!("0-{}", scope.scope_name()))
        .join(FORK_DB_FILE_NAME)
}

/// Opens or creates `fork.sqlite`, falling back to an in-memory database when
/// `ZED_STATELESS` is set or the file cannot be opened. Unlike `open_db`, a
/// file that fails to open is left in place rather than moved aside, so the
/// data stays available for inspection.
pub async fn open_fork_db(db_dir: &Path, scope: impl DbScope) -> ThreadSafeConnection {
    if !*ZED_STATELESS {
        let db_path = fork_db_path(db_dir, scope);
        let connection = maybe!(async {
            if let Some(parent) = db_path.parent() {
                create_dir_all(parent)
                    .context("Could not create fork db directory")
                    .log_err()?;
            }
            open_main_db::<ForkDb>(&db_path).await
        })
        .await;

        if let Some(connection) = connection {
            return connection;
        }
        log::error!(
            "Could not open {}; fork state will not persist across restarts",
            db_path.display()
        );
    }

    ThreadSafeConnection::builder::<ForkDb>(FORK_FALLBACK_DB_NAME, false)
        .with_db_initialization_query(DB_INITIALIZE_QUERY)
        .with_connection_initialize_query(CONNECTION_INITIALIZE_QUERY)
        .build()
        .await
        .expect(
            "Fallback in-memory fork database failed. Likely initialization queries or migrations have fundamental errors",
        )
}
