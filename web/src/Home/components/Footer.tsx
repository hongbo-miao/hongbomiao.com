import HmCopyright from '@/App/components/Copyright';
import HmSocialList from '@/App/components/SocialList';
import WEBSITES from '@/Home/fixtures/WEBSITES';

function Footer() {
  const year = new Date().getFullYear();
  return (
    <footer className="bg-background pb-16 pt-0">
      <div className="mx-auto max-w-[640px] px-6">
        <nav className="flex flex-col items-center justify-between gap-12 md:flex-row md:gap-0">
          <div className="flex flex-col items-center gap-4 md:flex-row">
            <HmSocialList websites={WEBSITES} />
          </div>
          <div className="flex items-center">
            <HmCopyright year={year} />
          </div>
        </nav>
      </div>
    </footer>
  );
}

export default Footer;
