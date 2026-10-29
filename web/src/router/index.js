import { createRouter, createWebHistory } from 'vue-router'
import i18n from '@/i18n'
import Layout from '@/layout/index.vue'
import { isLoggedIn } from '@/utils/auth'

const t = (key) => i18n.global.t(key)

export const menuRoutes = [
  { path: 'dashboard', name: 'Dashboard', component: () => import('@/views/dashboard/index.vue'), meta: { titleKey: 'menu.dashboard', icon: 'Odometer' } },
  { path: 'dataset', name: 'Dataset', component: () => import('@/views/dataset/index.vue'), meta: { titleKey: 'menu.dataset', icon: 'Coin' } },
  { path: 'pipeline', name: 'Pipeline', component: () => import('@/views/pipeline/index.vue'), meta: { titleKey: 'menu.pipeline', icon: 'MagicStick' } },
  { path: 'protocol', name: 'Protocol', component: () => import('@/views/protocol/index.vue'), meta: { titleKey: 'menu.protocol', icon: 'Grid' } },
  { path: 'arena', name: 'Arena', component: () => import('@/views/arena/index.vue'), meta: { titleKey: 'menu.arena', icon: 'Trophy' } },
  { path: 'hpo', name: 'Hpo', component: () => import('@/views/hpo/index.vue'), meta: { titleKey: 'menu.hpo', icon: 'Aim' } },
  { path: 'codex', name: 'Codex', component: () => import('@/views/codex/index.vue'), meta: { titleKey: 'menu.codex', icon: 'Collection' } },
  { path: 'plugins', name: 'Plugins', component: () => import('@/views/plugins/index.vue'), meta: { titleKey: 'menu.plugins', icon: 'Shop' } },
  { path: 'report', name: 'Report', component: () => import('@/views/report/index.vue'), meta: { titleKey: 'menu.report', icon: 'Document' } },
  { path: 'settings', name: 'Settings', component: () => import('@/views/settings/index.vue'), meta: { titleKey: 'menu.settings', icon: 'Setting' } },
]

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/login',
      name: 'Login',
      component: () => import('@/views/login/index.vue'),
      meta: { titleKey: 'menu.login', public: true },
    },
    { path: '/', component: Layout, redirect: '/dashboard', children: menuRoutes },
    { path: '/:pathMatch(.*)*', redirect: '/dashboard' },
  ],
})

router.beforeEach((to) => {
  if (to.meta.public) {
    return isLoggedIn() ? '/dashboard' : true
  }
  if (!isLoggedIn()) {
    return {
      path: '/login',
      query: to.fullPath === '/dashboard' ? {} : { redirect: to.fullPath },
    }
  }
  return true
})

router.afterEach((to) => {
  document.title = to.meta.titleKey ? `${t(to.meta.titleKey)} · NeuroKiln 脑炉` : 'NeuroKiln 脑炉'
})

export default router
