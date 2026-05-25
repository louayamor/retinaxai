# Role-Based Authentication — Implementation Plan

## Overview

Split current flat auth into three role-exclusive dashboards:

| Role | Dashboard | Pages |
|------|-----------|-------|
| `doctor` | `/dashboard/clinical/*` | patients, predictions, reports, AI chat, analytics |
| `engineer` | `/dashboard/engineering/*` | models, mlops, llmops, system stats |
| `admin` | `/dashboard/admin/*` | user management, settings, platform stats |

No role overlap. Admin cannot access clinical or engineering pages. Admin seeds accounts — no public registration.

## PR Dependency Graph

```
PR 1 ──→ PR 2 ──→ PR 3 ──→ PR 4 ──→ PR 5
```

Each PR is independently creatable (draft). Only merge order matters.

---

## PR 1 — Backend: Role Column + Model + Enum

**Branch:** `feat/backend-role-column`

Safe additive — no behavioral change. Adds `role` field everywhere but nothing reads it yet.

| File | Change |
|------|--------|
| `app/auth/roles.py` | NEW — `Role(str, Enum): DOCTOR, ENGINEER, ADMIN` |
| `app/models/user.py` | Add `role: Mapped[Role]` column (String(20)) |
| `app/db/migrations/versions/..._add_role_column.py` | NEW — `ALTER TABLE users ADD role VARCHAR(20) DEFAULT 'doctor' NOT NULL` |
| `app/schemas/user_schema.py` | Add `role` to `UserCreate` (default doctor), `UserRead`, `UserUpdate` |
| `app/users/service.py` | Pass `role` from schema in `create()`; add `list()` method |
| `app/users/repository.py` | Add `list_all()`, `count()` |

---

## PR 2 — Backend: Role Guard + Admin Routes + Register Protection

**Branch:** `feat/backend-rbac`

Enables role enforcement, admin management endpoints, and admin seed user.

| File | Change |
|------|--------|
| `app/auth/role_guard.py` | NEW — `require_role()`, `DoctorUser`, `EngineerUser`, `AdminUser` |
| Clinical routes (patients, predictions, reports, scans, chat, explanations, dashboard) | `CurrentUser` → `DoctorUser` |
| Engineering routes (metrics, system, orchestration) | `CurrentUser` → `EngineerUser` |
| `app/api/v1/routes/admin_routes.py` | NEW — `GET /admin/users`, `PATCH /admin/users/{id}`, `GET /admin/stats` |
| `app/api/v1/routes/auth_routes.py` | Protect `/register` with `AdminUser`; add `role` to responses |
| `scripts/seed_admin.py` | NEW — env-based admin user creation |

---

## PR 3 — Frontend: Auth Context + Role-Based Redirect

**Branch:** `feat/frontend-auth-context`

Prepares frontend to know the user's role on every page.

| File | Change |
|------|--------|
| `src/lib/auth.ts` | Add `role` to `AuthUser` type |
| `src/providers/auth-context.tsx` | NEW — React context: `user`, `role`, `loading`, `logout()` |
| `src/providers/providers.tsx` | Wrap children in `AuthProvider` |
| `src/app/auth/login/page.tsx` | Redirect based on role after login |
| `src/features/auth/components/user-auth-form.tsx` | Remove register form/tab |
| `src/app/auth/register/page.tsx` | DELETE |

---

## PR 4 — Frontend: Route Restructure + Role-Based Sidebar

**Branch:** `feat/frontend-role-routes`

Moves files into role-specific route groups, updates sidebar to render role-appropriate nav.

| Move | From → To |
|------|-----------|
| Patients | `/dashboard/patients/*` → `/dashboard/clinical/patients/*` |
| Predictions | `/dashboard/predictions/*` → `/dashboard/clinical/predictions/*` |
| Reports | `/dashboard/reports/*` → `/dashboard/clinical/reports/*` |
| Chat | `/dashboard/chat/*` → `/dashboard/clinical/chat/*` |
| Analytics | `/dashboard/analytics/*` → `/dashboard/clinical/analytics/*` |
| Overview | `/dashboard/overview/*` → `/dashboard/clinical/page.tsx` |
| Models | `/dashboard/models/*` → `/dashboard/engineering/models/*` |
| MLOps | `/dashboard/mlops/*` → `/dashboard/engineering/mlops/*` |
| LLMOps | `/dashboard/llmops/*` → `/dashboard/engineering/llmops/*` |
| System | `/dashboard/system/*` → `/dashboard/engineering/system/*` |
| Engineering overview | NEW → `/dashboard/engineering/page.tsx` |
| Admin overview | NEW → `/dashboard/admin/page.tsx` |
| Profile | stays at `/dashboard/profile/` (shared) |

`nav-config.ts` → split into `clinicalNav`, `engineeringNav`, `adminNav`.
`app-sidebar.tsx` → reads role from `useAuth()`, renders appropriate nav.

---

## PR 5 — Frontend: Admin User Management

**Branch:** `feat/admin-user-management`

| File | Description |
|------|-------------|
| `/dashboard/admin/users/page.tsx` | NEW — data table: username, email, role badge, active toggle |
| `/dashboard/admin/users/[id]/page.tsx` | NEW — user detail/edit view |
| `/dashboard/admin/settings/page.tsx` | NEW — app settings placeholder |
| `src/lib/api.ts` | Add `listUsers()`, `updateUser()`, `getAdminStats()` |

---

## Key Design Decisions

1. **Strict silos** — no cross-role access. Admin cannot view clinical or engineering pages.
2. **Admin seeds accounts** — no public registration. `POST /auth/register` is admin-only.
3. **Upward access** — NOT implemented. Each role is exclusive to its domain.
4. **Token contains `role`** — JWT has `role` claim so frontend can redirect without extra API call after login. Survives page refresh (hydrate from `/auth/me`).
5. **File moves** — existing page files physically move into new route group directories (loses git blame history).
6. **Redirects** — old `/dashboard/patients` etc. get redirect pages or `next.config.js` entries to new paths.
