<template>
  <header class="navbar">
    <div class="navbar-left">
      <el-icon class="hamburger" @click="$emit('toggle')">
        <Fold v-if="!collapsed" /><Expand v-else />
      </el-icon>
      <el-breadcrumb separator="/">
        <el-breadcrumb-item :to="{ path: '/dashboard' }">{{ $t('navbar.home') }}</el-breadcrumb-item>
        <el-breadcrumb-item v-if="$route.path !== '/dashboard'">{{ $t($route.meta.titleKey) }}</el-breadcrumb-item>
      </el-breadcrumb>
    </div>
    <div class="navbar-right">
      <el-dropdown @command="onLocale">
        <span class="lang-switch" :title="$t('navbar.language')">
          {{ locale === 'zh' ? '中文' : 'EN' }}
        </span>
        <template #dropdown>
          <el-dropdown-menu>
            <el-dropdown-item command="zh" :disabled="locale === 'zh'">中文</el-dropdown-item>
            <el-dropdown-item command="en" :disabled="locale === 'en'">English</el-dropdown-item>
          </el-dropdown-menu>
        </template>
      </el-dropdown>
      <el-tooltip :content="$t('navbar.refresh')"><el-icon class="action" @click="$emit('refresh')"><Refresh /></el-icon></el-tooltip>
      <el-tooltip :content="$t('navbar.docs')" :show-after="200">
        <el-icon class="action" @click="openDocs"><Link /></el-icon>
      </el-tooltip>
      <el-dropdown @command="onCommand">
        <span class="user">
          <el-avatar :size="28" class="avatar">{{ initial }}</el-avatar>
          <span>{{ displayName }}</span>
          <el-tag v-if="admin" size="small" type="warning" effect="plain">{{ $t('navbar.admin') }}</el-tag>
          <el-icon><ArrowDown /></el-icon>
        </span>
        <template #dropdown>
          <el-dropdown-menu>
            <el-dropdown-item command="settings">{{ $t('navbar.settings') }}</el-dropdown-item>
            <el-dropdown-item command="about">{{ $t('navbar.about') }}</el-dropdown-item>
            <el-dropdown-item divided command="logout">{{ $t('navbar.logout') }}</el-dropdown-item>
          </el-dropdown-menu>
        </template>
      </el-dropdown>
    </div>
  </header>
</template>

<script setup>
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import { ElMessageBox } from 'element-plus'
import { setLocale } from '@/i18n'
import { authState, clearSession, isAdmin } from '@/utils/auth'

defineProps({ collapsed: Boolean })
defineEmits(['toggle', 'refresh'])

const route = useRoute()
const router = useRouter()
const { t, locale } = useI18n()
const displayName = computed(() => authState.user?.username || t('navbar.researcher'))
const initial = computed(() => (displayName.value || 'N').slice(0, 1).toUpperCase())
const admin = computed(() => isAdmin())

function onLocale(command) {
  if (command === locale.value) return
  setLocale(command)
  if (route.meta.titleKey) {
    document.title = `${t(route.meta.titleKey)} · NeuroKiln 脑炉`
  }
}

function openDocs() {
  window.open('http://127.0.0.1:8100/docs', '_blank')
}

function about() {
  ElMessageBox.alert(
    t('navbar.aboutText'),
    t('navbar.aboutTitle'),
    { confirmButtonText: t('navbar.understood') },
  )
}

async function logout() {
  try {
    await ElMessageBox.confirm(t('navbar.logoutConfirm'), t('navbar.logoutTitle'), {
      type: 'warning', confirmButtonText: t('navbar.logoutButton'), cancelButtonText: t('common.cancel'),
    })
  } catch {
    return
  }
  clearSession()
  await router.replace('/login')
}

function onCommand(command) {
  if (command === 'settings') router.push('/settings')
  else if (command === 'about') about()
  else if (command === 'logout') logout()
}
</script>

<style scoped>
.navbar {
  height: var(--navbar-height); background: #fff; display: flex; align-items: center;
  justify-content: space-between; padding: 0 16px; box-shadow: 0 1px 4px rgba(0,21,41,.08);
  z-index: 1000;
}
.navbar-left { display: flex; align-items: center; gap: 14px; }
.hamburger { font-size: 20px; cursor: pointer; color: #5a5e66; }
.navbar-right { display: flex; align-items: center; gap: 16px; }
.action { font-size: 18px; color: #5a5e66; cursor: pointer; }
.lang-switch {
  font-size: 13px; color: #5a5e66; cursor: pointer; padding: 3px 9px;
  border: 1px solid #dcdfe6; border-radius: 10px; line-height: 1;
}
.lang-switch:hover { color: #409eff; border-color: #409eff; }
.user { display: flex; align-items: center; gap: 8px; cursor: pointer; color: #606266; }
.avatar { background: #409eff; }
</style>
