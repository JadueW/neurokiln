import { defineConfig } from 'vitepress'

const base = process.env.DOCS_BASE || '/'

export default defineConfig({
  lang: 'zh-CN',
  title: 'NeuroKiln',
  description: '可复现、防泄漏的 BCI 算法评测与调参平台',
  base,
  cleanUrls: true,
  head: [
    ['link', { rel: 'icon', type: 'image/png', href: `${base}logo.png` }],
    ['meta', { name: 'theme-color', content: '#0b1220' }],
  ],
  themeConfig: {
    logo: '/logo.png',
    nav: [
      { text: '用户手册', link: '/user-manual' },
      { text: '在线闭环', link: '/online-closed-loop' },
      { text: '安全模型', link: '/security' },
    ],
    sidebar: [
      {
        text: '开始',
        items: [
          { text: '用户手册', link: '/user-manual' },
          { text: '在线与闭环扩展', link: '/online-closed-loop' },
          { text: '安全模型', link: '/security' },
        ],
      },
    ],
    outline: { level: [2, 3], label: '本页目录' },
    search: { provider: 'local' },
    docFooter: { prev: '上一节', next: '下一节' },
    footer: {
      message: 'Apache-2.0 · 仅用于科研，不构成医疗建议',
      copyright: 'Copyright 2026 NeuroKiln Contributors',
    },
  },
})
